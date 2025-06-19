# tests/test_context_optimizer.py

from utils.context_optimizer import optimize_context

def test_optimize_context():
    text = "Break a leg!"
    optimized = optimize_context(text, src_lang="en", tgt_lang="zh")
    assert "祝你好运" in optimized
