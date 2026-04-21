"""Tests for gp1.lm.train_kenlm.

TDD: RED phase — these tests were written before any implementation.

Key constraints verified here:
- FileNotFoundError with a helpful message when lmplz is absent from PATH.
- subprocess.run is called twice (lmplz + build_binary) with the correct args.
- Intermediate .arpa file is cleaned up after a successful run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Skip the whole module at collection time if the implementation is missing
# (pre-GREEN). pytest.skip(..., allow_module_level=True) is NoReturn so that
# Pyright can type-narrow subsequent references to train_kenlm.
# ---------------------------------------------------------------------------
try:
    from gp1.lm.train_kenlm import train_kenlm
except ImportError:
    pytest.skip(
        "gp1.lm.train_kenlm not yet implemented (RED phase expected)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_corpus(tmp_path: Path) -> Path:
    """A tiny corpus file that stands in for the real one."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("тысяча\nдве тысячи\n", encoding="utf-8")
    return corpus


@pytest.fixture()
def fake_binary_out(tmp_path: Path) -> Path:
    """Destination path for the .bin output."""
    return tmp_path / "lm.bin"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRaisesFileNotFoundWhenLmplzMissing:
    """test_raises_file_not_found_when_lmplz_missing"""

    def test_raises_file_not_found_when_lmplz_missing(
        self, fake_corpus: Path, fake_binary_out: Path
    ) -> None:
        """
        Arrange: shutil.which returns None for lmplz (simulating absent binary).
        Act: call train_kenlm.
        Assert: FileNotFoundError is raised with a message mentioning lmplz.
        """
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError) as exc_info:
                train_kenlm(
                    corpus_path=fake_corpus,
                    out_binary=fake_binary_out,
                )

        msg = str(exc_info.value)
        assert "lmplz" in msg.lower(), (
            f"Error message should mention 'lmplz', got: {msg!r}"
        )
        assert "path" in msg.lower() or "install" in msg.lower(), (
            f"Error message should mention PATH or install instructions, got: {msg!r}"
        )


class TestInvokesLmplzAndBuildBinary:
    """test_invokes_lmplz_and_build_binary"""

    def test_invokes_lmplz_and_build_binary(
        self, fake_corpus: Path, fake_binary_out: Path
    ) -> None:
        """
        Arrange: shutil.which returns a dummy path for lmplz/build_binary;
                 subprocess.run is mocked to avoid real shell calls.
        Act: call train_kenlm(order=4).
        Assert: subprocess.run is called exactly twice — first for lmplz with
                --discount_fallback flag, then for build_binary with 'trie'.
        """

        def _which_side_effect(cmd: str) -> str | None:
            return f"/usr/local/bin/{cmd}"

        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=b""))

        with (
            patch("shutil.which", side_effect=_which_side_effect),
            patch("subprocess.run", mock_run),
        ):
            train_kenlm(
                corpus_path=fake_corpus,
                out_binary=fake_binary_out,
                order=4,
            )

        assert mock_run.call_count == 2, (
            f"subprocess.run should be called exactly twice, got {mock_run.call_count}"
        )

        # First call: lmplz
        first_call_args = mock_run.call_args_list[0]
        cmd_list = first_call_args[0][0]  # positional arg 0 is the command list
        assert cmd_list[0] == "/usr/local/bin/lmplz" or "lmplz" in cmd_list[0], (
            f"First subprocess call should be lmplz, got: {cmd_list}"
        )
        assert "-o" in cmd_list, f"lmplz call missing -o flag: {cmd_list}"
        assert "4" in [str(x) for x in cmd_list], (
            f"lmplz call missing order=4: {cmd_list}"
        )
        assert "--discount_fallback" in cmd_list, (
            f"lmplz call missing --discount_fallback: {cmd_list}"
        )

        # Second call: build_binary
        second_call_args = mock_run.call_args_list[1]
        cmd_list2 = second_call_args[0][0]
        assert "build_binary" in cmd_list2[0] or any(
            "build_binary" in str(x) for x in cmd_list2
        ), f"Second subprocess call should be build_binary, got: {cmd_list2}"
        assert "trie" in cmd_list2, (
            f"build_binary call missing 'trie' argument: {cmd_list2}"
        )

    def test_passes_vocab_limit_file_when_provided(
        self, fake_corpus: Path, fake_binary_out: Path, tmp_path: Path
    ) -> None:
        """
        Arrange: vocab_limit_path is provided.
        Assert: lmplz command includes --limit_vocab_file <path>.
        """
        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("тысяча\n", encoding="utf-8")

        def _which_side_effect(cmd: str) -> str | None:
            return f"/usr/local/bin/{cmd}"

        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=b""))

        with (
            patch("shutil.which", side_effect=_which_side_effect),
            patch("subprocess.run", mock_run),
        ):
            train_kenlm(
                corpus_path=fake_corpus,
                out_binary=fake_binary_out,
                order=4,
                vocab_limit_path=vocab_path,
            )

        first_call_args = mock_run.call_args_list[0]
        cmd_list = first_call_args[0][0]
        assert "--limit_vocab_file" in cmd_list, (
            f"lmplz call missing --limit_vocab_file: {cmd_list}"
        )


class TestCleansUpIntermediateArpa:
    """test_cleans_up_intermediate_arpa"""

    def test_cleans_up_intermediate_arpa(
        self, fake_corpus: Path, fake_binary_out: Path
    ) -> None:
        """
        Arrange: subprocess.run is mocked (no real kenlm needed).
        Act: call train_kenlm.
        Assert: no .arpa file remains in the output directory after the call.
        """
        out_dir = fake_binary_out.parent

        def _which_side_effect(cmd: str) -> str | None:
            return f"/usr/local/bin/{cmd}"

        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=b""))

        with (
            patch("shutil.which", side_effect=_which_side_effect),
            patch("subprocess.run", mock_run),
        ):
            train_kenlm(
                corpus_path=fake_corpus,
                out_binary=fake_binary_out,
            )

        leftover_arpa = list(out_dir.glob("*.arpa"))
        assert not leftover_arpa, (
            f"Intermediate .arpa files not cleaned up: {leftover_arpa}"
        )
