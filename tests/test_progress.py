from foundry.progress import _format, emit, set_verbose


def test_format_includes_step_and_truncates_long_text():
    line = _format(
        {
            "step": "founder.response",
            "run_id": "abcdefgh-1234",
            "response": "x" * 500,
        }
    )
    assert "[foundry:abcdefgh]" in line
    assert "founder.response" in line
    assert "…" in line


def test_emit_quiet_skips_print(capsys):
    set_verbose(False)
    emit("run.start", run_id="rid", prompt="hi")
    captured = capsys.readouterr()
    assert captured.out == ""
    set_verbose(True)


def test_emit_verbose_prints(capsys):
    set_verbose(True)
    emit("categorize.decision", run_id="abcdef01-xxxx", route="founder", confidence=0.9)
    captured = capsys.readouterr()
    assert "categorize.decision" in captured.out
    assert "route=founder" in captured.out
