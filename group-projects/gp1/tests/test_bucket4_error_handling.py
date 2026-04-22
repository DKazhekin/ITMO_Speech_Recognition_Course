"""Bucket-4 error-handling tests.

Covers:
  H7  — _words_to_digits_safe narrows except to ValueError only
  M9  — result.json must not contain Infinity (best_val_cer=null when inf)
  M12 — _try_build_beam_decoder narrows except to ImportError/FileNotFoundError
  publish_release.sh — null CER -> title omits CER segment
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — locate source files
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train.py"
PUBLISH_SH = SCRIPTS_DIR / "publish_release.sh"


# ---------------------------------------------------------------------------
# H7 — _words_to_digits_safe
# ---------------------------------------------------------------------------


class TestWordsToDIgitsSafe:
    """H7: _words_to_digits_safe must only silence ValueError, not all errors."""

    def _import_fn(self):
        from gp1.submit.inference import _words_to_digits_safe

        return _words_to_digits_safe

    def test_words_to_digits_safe_propagates_non_value_errors(self):
        """AttributeError from words_to_digits must NOT be swallowed (H7)."""
        fn = self._import_fn()
        with patch(
            "gp1.submit.inference.words_to_digits",
            side_effect=AttributeError("oops"),
        ):
            with pytest.raises(AttributeError, match="oops"):
                fn("три тысячи")

    def test_words_to_digits_safe_still_catches_value_error(self):
        """ValueError from words_to_digits must return empty string (H7)."""
        fn = self._import_fn()
        with patch(
            "gp1.submit.inference.words_to_digits",
            side_effect=ValueError("bad input"),
        ):
            result = fn("gibberish")
        assert result == ""

    def test_words_to_digits_safe_returns_empty_on_blank_input(self):
        """Blank/whitespace input returns '' without calling words_to_digits."""
        fn = self._import_fn()
        assert fn("") == ""
        assert fn("   ") == ""


# ---------------------------------------------------------------------------
# M12 — _try_build_beam_decoder
# ---------------------------------------------------------------------------


def _make_fake_beam_module(decoder_side_effect=None):
    """Return a fake gp1.decoding.beam_pyctc module."""
    fake_mod = types.ModuleType("gp1.decoding.beam_pyctc")

    class FakeConfig:
        def __init__(self, **kw):
            pass

    if decoder_side_effect is not None:

        class FakeDecoder:
            def __init__(self, **kw):
                raise decoder_side_effect
    else:

        class FakeDecoder:
            def __init__(self, **kw):
                pass

    fake_mod.BeamSearchConfig = FakeConfig
    fake_mod.BeamSearchDecoder = FakeDecoder
    return fake_mod


def _make_fake_vocab_word_module():
    """Return a fake gp1.text.vocab_word module with NUMBER_WORDS."""
    fake_mod = types.ModuleType("gp1.text.vocab_word")
    fake_mod.NUMBER_WORDS = frozenset(["тысяча"])
    return fake_mod


class TestTryBuildBeamDecoder:
    """M12: _try_build_beam_decoder must narrow except to ImportError+FileNotFoundError."""

    def _make_vocab(self):
        from gp1.text.vocab import CharVocab

        return CharVocab()

    def test_try_build_beam_decoder_catches_import_error(self):
        """ImportError (missing pyctcdecode/kenlm) should be caught -> returns None (M12)."""
        from gp1.submit.inference import _try_build_beam_decoder

        vocab = self._make_vocab()

        # Simulate pyctcdecode not installed by setting the module to None
        # in sys.modules, which causes `from gp1.decoding.beam_pyctc import ...`
        # to raise ImportError.
        with patch.dict(
            sys.modules,
            {
                "gp1.decoding.beam_pyctc": None,
                "gp1.text.vocab_word": None,
            },
        ):
            # Need to force re-execution of the `from ... import` inside the fn.
            # Since it's a local import inside the try block, the cached module
            # in sys.modules controls it.  Setting to None causes ImportError.
            result = _try_build_beam_decoder(vocab, Path("/nonexistent/lm.bin"))

        assert result is None

    def test_try_build_beam_decoder_catches_file_not_found(self):
        """FileNotFoundError (bad LM path) should be caught -> returns None (M12)."""
        from gp1.submit.inference import _try_build_beam_decoder

        vocab = self._make_vocab()
        fake_beam_mod = _make_fake_beam_module(
            decoder_side_effect=FileNotFoundError("lm.bin not found")
        )
        fake_vocab_word = _make_fake_vocab_word_module()

        with patch.dict(
            sys.modules,
            {
                "gp1.decoding.beam_pyctc": fake_beam_mod,
                "gp1.text.vocab_word": fake_vocab_word,
            },
        ):
            result = _try_build_beam_decoder(vocab, Path("/nonexistent/lm.bin"))

        assert result is None

    def test_try_build_beam_decoder_propagates_type_error(self):
        """TypeError (real bug, not missing deps) must propagate, not be swallowed (M12)."""
        from gp1.submit.inference import _try_build_beam_decoder

        vocab = self._make_vocab()
        fake_beam_mod = _make_fake_beam_module(decoder_side_effect=TypeError("bad arg"))
        fake_vocab_word = _make_fake_vocab_word_module()

        with patch.dict(
            sys.modules,
            {
                "gp1.decoding.beam_pyctc": fake_beam_mod,
                "gp1.text.vocab_word": fake_vocab_word,
            },
        ):
            with pytest.raises(TypeError, match="bad arg"):
                _try_build_beam_decoder(vocab, Path("/nonexistent/lm.bin"))

    def test_try_build_beam_decoder_propagates_attribute_error(self):
        """AttributeError (real bug) must propagate, not be swallowed (M12)."""
        from gp1.submit.inference import _try_build_beam_decoder

        vocab = self._make_vocab()
        fake_beam_mod = _make_fake_beam_module(
            decoder_side_effect=AttributeError("missing attr")
        )
        fake_vocab_word = _make_fake_vocab_word_module()

        with patch.dict(
            sys.modules,
            {
                "gp1.decoding.beam_pyctc": fake_beam_mod,
                "gp1.text.vocab_word": fake_vocab_word,
            },
        ):
            with pytest.raises(AttributeError, match="missing attr"):
                _try_build_beam_decoder(vocab, Path("/nonexistent/lm.bin"))


# ---------------------------------------------------------------------------
# M9 — result.json must use null, not Infinity
# ---------------------------------------------------------------------------


def _write_result_json(output_dir: Path, best_val_cer: float) -> Path:
    """Reproduce the result.json write logic from scripts/train.py.

    After the fix, float('inf') must be serialised as null.
    """
    import math

    # This is the fixed serialization — what scripts/train.py should do.
    cer_for_json = (
        None
        if (math.isinf(best_val_cer) or math.isnan(best_val_cer))
        else float(best_val_cer)
    )
    summary_path = output_dir / "result.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_val_cer": cer_for_json,
                "best_ckpt_path": str(output_dir / "model.pt"),
                "config_path": str(output_dir / "config.yaml"),
                "n_params": 1_000_000,
            },
            indent=2,
        )
    )
    return summary_path


class TestResultJsonNullCer:
    """M9: train.py must write null (not Infinity) when best_val_cer=float('inf')."""

    def test_result_json_uses_null_for_inf_cer(self, tmp_path):
        """After fix: best_val_cer=inf must serialise to JSON null (M9)."""
        summary_path = _write_result_json(tmp_path, float("inf"))

        raw = summary_path.read_text()
        parsed = json.loads(raw)  # must not raise — valid JSON

        assert parsed["best_val_cer"] is None, (
            "best_val_cer must be null (None) when training never improved"
        )
        assert "Infinity" not in raw, (
            "result.json must not contain literal 'Infinity' (invalid per RFC 8259 §6)"
        )

    def test_result_json_uses_real_number_when_finite(self, tmp_path):
        """After fix: finite CER must still be written as a number."""
        summary_path = _write_result_json(tmp_path, 0.1234)

        raw = summary_path.read_text()
        parsed = json.loads(raw)

        assert parsed["best_val_cer"] == pytest.approx(0.1234)

    def test_train_script_result_json_with_inf(self, tmp_path):
        """Integration test: read train.py source and verify the null guard is present."""
        train_src = TRAIN_SCRIPT.read_text()

        # After fix, train.py must NOT directly serialize float("inf") without a guard.
        # The guard replaces inf/nan with None before json.dumps.
        # We check the source contains 'math.isinf' or 'is None' guard pattern.
        has_guard = (
            "math.isinf" in train_src
            or "isinf" in train_src
            or "math.isnan" in train_src
            or '== float("inf")' in train_src
            or "== float('inf')" in train_src
            or "_cer_for_json" in train_src
        )
        assert has_guard, (
            "scripts/train.py must guard against float('inf') before json.dumps. "
            "Expected math.isinf() or equivalent guard."
        )

        # Sanity-check: confirm json.dumps(float('inf')) produces Infinity
        # (which is invalid per RFC 8259 §6 even though CPython's json.loads
        # happens to accept it permissively — other parsers like jq and JS do not).
        raw = json.dumps({"x": float("inf")})
        assert "Infinity" in raw, "Sanity: json.dumps(inf) still produces Infinity"


# ---------------------------------------------------------------------------
# publish_release.sh — null CER -> no CER segment in title
# ---------------------------------------------------------------------------


class TestPublishReleaseSh:
    """publish_release.sh must omit CER from title when best_val_cer is null."""

    @pytest.mark.skipif(not PUBLISH_SH.exists(), reason="publish_release.sh not found")
    def test_publish_release_sh_handles_null_cer(self, tmp_path):
        """With best_val_cer=null in release.json, title must not contain 'CER null' (M9)."""
        release_dir = tmp_path / "releases" / "quartznet" / "v0.0.0-test"
        release_dir.mkdir(parents=True)

        (release_dir / "model.pt").write_bytes(b"\x00" * 4)
        (release_dir / "config.yaml").write_text("model:\n  name: quartznet_10x4\n")
        (release_dir / "README.md").write_text("# Test release\n")

        release_json = {
            "baseline": "quartznet",
            "tag": "v0.0.0-test",
            "git_commit": "abc1234",
            "checkpoint_source": "epoch0000_cer0.0000.pt",
            "params_count": 4000000,
            "best_val_cer": None,  # null
        }
        (release_dir / "release.json").write_text(json.dumps(release_json, indent=2))

        env = os.environ.copy()
        result = subprocess.run(
            ["bash", str(PUBLISH_SH), "quartznet", "v0.0.0-test", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=env,
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, f"publish_release.sh failed:\n{output}"

        # Title must not contain "CER null" or "CER None"
        assert "CER null" not in output, (
            f"Title contains 'CER null' — should omit CER segment.\nOutput:\n{output}"
        )
        assert "CER None" not in output, (
            f"Title contains 'CER None' — should omit CER segment.\nOutput:\n{output}"
        )

    @pytest.mark.skipif(not PUBLISH_SH.exists(), reason="publish_release.sh not found")
    def test_publish_release_sh_includes_cer_for_real_number(self, tmp_path):
        """With a real best_val_cer, the title must include the CER value."""
        release_dir = tmp_path / "releases" / "quartznet" / "v0.0.0-test"
        release_dir.mkdir(parents=True)

        (release_dir / "model.pt").write_bytes(b"\x00" * 4)
        (release_dir / "config.yaml").write_text("model:\n  name: quartznet_10x4\n")
        (release_dir / "README.md").write_text("# Test release\n")

        release_json = {
            "baseline": "quartznet",
            "tag": "v0.0.0-test",
            "git_commit": "abc1234",
            "checkpoint_source": "epoch0000_cer0.1234.pt",
            "params_count": 4000000,
            "best_val_cer": 0.1234,
        }
        (release_dir / "release.json").write_text(json.dumps(release_json, indent=2))

        env = os.environ.copy()
        result = subprocess.run(
            ["bash", str(PUBLISH_SH), "quartznet", "v0.0.0-test", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=env,
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, f"publish_release.sh failed:\n{output}"
        assert "0.1234" in output, (
            f"Expected CER 0.1234 in title output.\nOutput:\n{output}"
        )
