"""Tests for CustomCodeGenerator AST sandbox security."""
import pytest
import pandas as pd
import numpy as np

from quantlab.factor_discovery.factor_enhancements import CustomCodeGenerator


@pytest.fixture
def generator():
    return CustomCodeGenerator()


@pytest.fixture
def market_df():
    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    assets = [f"{i:06d}" for i in range(1, 21)]
    rows = []
    for d in dates:
        for a in assets:
            rows.append({
                "date": d, "asset": a,
                "close": np.random.uniform(10, 100),
                "high": np.random.uniform(10, 100),
                "low": np.random.uniform(5, 95),
                "volume": np.random.uniform(1e5, 1e7),
            })
    return pd.DataFrame(rows)


class TestASTValidation:
    def test_legitimate_factor_passes(self, generator):
        """Valid factor code passes AST check."""
        code = """\
def compute_factor(df):
    return df.groupby('asset')['close'].transform(lambda x: x.pct_change(5).rank())
"""
        result = generator._ast_validate(code)
        assert result == [], f"Legitimate code should pass: {result}"

    def test_dangerous_import_rejected(self, generator):
        """AST catches import os."""
        code = """\
import os
def compute_factor(df):
    os.system('evil')
    return df['close']
"""
        result = generator._ast_validate(code)
        assert any("导入" in r for r in result) or any("import" in r.lower() for r in result), f"Should catch import os: {result}"

    def test_from_import_rejected(self, generator):
        """AST catches from subprocess import."""
        code = """\
from subprocess import check_output
def compute_factor(df):
    return df['close']
"""
        result = generator._ast_validate(code)
        assert any("导入" in r for r in result) or any("import" in r.lower() for r in result), f"Should catch from import: {result}"

    def test_eval_call_rejected(self, generator):
        """AST catches eval() call."""
        code = """\
def compute_factor(df):
    eval('print(1)')
    return df['close']
"""
        result = generator._ast_validate(code)
        assert len(result) > 0, f"Should catch eval: {result}"

    def test_exec_call_rejected(self, generator):
        """AST catches exec() call."""
        code = """\
def compute_factor(df):
    exec('x = 1')
    return df['close']
"""
        result = generator._ast_validate(code)
        assert len(result) > 0, f"Should catch exec: {result}"

    def test_dunder_class_access_rejected(self, generator):
        """AST catches __class__ attribute access."""
        code = """\
def compute_factor(df):
    x = df.__class__.__mro__
    return df['close']
"""
        result = generator._ast_validate(code)
        assert any("__class__" in r for r in result), f"Should catch __class__: {result}"

    def test_dunder_globals_rejected(self, generator):
        """AST catches __globals__ dunder access."""
        code = """\
def compute_factor(df):
    g = compute_factor.__globals__
    return df['close']
"""
        result = generator._ast_validate(code)
        assert any("__globals__" in r for r in result), f"Should catch __globals__: {result}"

    def test_open_call_rejected(self, generator):
        """AST catches open() call."""
        code = """\
def compute_factor(df):
    f = open('/etc/passwd')
    return df['close']
"""
        result = generator._ast_validate(code)
        assert len(result) > 0, f"Should catch open: {result}"

    def test_compile_call_rejected(self, generator):
        """AST catches compile() call."""
        code = """\
def compute_factor(df):
    c = compile('x=1', '', 'exec')
    return df['close']
"""
        result = generator._ast_validate(code)
        assert len(result) > 0, f"Should catch compile: {result}"

    def test_substring_bypass_caught_by_ast(self, generator):
        """String concatenation bypass of substring blacklist is caught by AST."""
        code = """\
def compute_factor(df):
    g = eval('op' + 'en')
    return df['close']
"""
        result = generator._ast_validate(code)
        assert len(result) > 0, f"AST should catch eval even with string concat: {result}"

    def test_numpy_pandas_import_allowed(self, generator):
        """Import pandas and numpy are allowed by AST."""
        code = """\
import pandas as pd
import numpy as np
def compute_factor(df):
    return pd.Series(np.random.randn(len(df)), index=df.index)
"""
        result = generator._ast_validate(code)
        assert result == [], f"pandas/numpy import should pass: {result}"

    def test_syntax_error_caught(self, generator):
        """Syntax errors are caught gracefully."""
        code = """\
def compute_factor(df
    return df['close']
"""
        result = generator._ast_validate(code)
        assert len(result) > 0, "Should catch syntax error"

    def test_builtins_subscript_rejected(self, generator):
        """AST catches __builtins__['eval'] access pattern."""
        code = """\
def compute_factor(df):
    b = __builtins__['eval']
    return df['close']
"""
        result = generator._ast_validate(code)
        assert any("__builtins__" in r for r in result), f"Should catch __builtins__ access: {result}"


class TestSafetyCheckIntegrated:
    def test_safety_check_ast_and_substring(self, generator):
        """_safety_check uses both AST and substring defenses."""
        code = """\
def compute_factor(df):
    return df.groupby('asset')['close'].transform('rank')
"""
        result = generator._safety_check(code)
        assert result["safe"], f"Safe code should pass: {result['reasons']}"

    def test_safety_check_rejects_dangerous(self, generator):
        """_safety_check rejects code with both eval() and import os."""
        code = """\
import os
def compute_factor(df):
    os.system('ls')
    return df['close']
"""
        result = generator._safety_check(code)
        assert not result["safe"], "Dangerous code should be rejected"
        assert len(result["reasons"]) > 1, "Should have multiple violation reasons"

    def test_safety_check_requires_compute_factor(self, generator):
        """_safety_check rejects code without compute_factor function."""
        code = "x = 1\nprint(x)"
        result = generator._safety_check(code)
        assert not result["safe"]
        assert any("compute_factor" in r for r in result["reasons"])

    def test_getattr_bypass_caught(self, generator):
        """getattr() call is still caught by substring blacklist."""
        code = """\
def compute_factor(df):
    g = getattr(df, '__class__')
    return df['close']
"""
        result = generator._safety_check(code)
        assert not result["safe"], f"getattr should be caught: {result['reasons']}"


class TestSandboxExecute:
    def test_valid_code_executes(self, generator, market_df):
        """Valid factor code executes and returns a Series."""
        code = """\
import pandas as pd
import numpy as np

def compute_factor(df):
    result = df.groupby('asset')['close'].transform(lambda x: x.pct_change(5))
    return result.rank(pct=True)
"""
        # Bypass safety for this verification test
        result = generator._sandbox_execute(code, market_df)
        assert result["status"] == "success", f"Expected success: {result.get('error', '')}"
        assert isinstance(result["factor_values"], pd.Series)

    def test_executed_returns_correct_length(self, generator, market_df):
        """Valid factor returns Series with same length as input."""
        code = """\
def compute_factor(df):
    return df['close'] / df['close'].mean() - 1
"""
        result = generator._sandbox_execute(code, market_df)
        assert result["status"] == "success"
        assert len(result["factor_values"]) == len(market_df)
