import ast
import json
from pathlib import Path
import unittest

import numpy as np

from run_ablation import build_candidate_matrices


NOTEBOOK = Path(__file__).resolve().parents[1] / "Model tham khảo" / "Grandmaster" / "notebook.ipynb"
MODELS = ("02e_SpGLM", "02f_DeGLM", "03a_OffLGB", "03b_OffXGB", "03c_OffCAT", "03d_StXGB", "04_EBM")


def archived_decomposition():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    definitions = ast.parse("".join(cells[7]["source"]))
    function = next(node for node in definitions.body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_resdecomp")
    scope = {"np": np, "LOCKED_MODELS": MODELS}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    return scope["build_resdecomp"]


class CandidateMatrixTests(unittest.TestCase):
    def test_alternatives_change_only_their_own_meta_input(self):
        logits = [np.array([value], dtype=float) for value in (-4, -2, -1, -2, -3, -4, -2)]
        baseline = dict(zip(MODELS, logits))
        raw_candidates = {
            "residual_te": np.array([-1.5]),
            "hierarchical_amount": np.array([-1.0]),
            "aligned_offset": np.array([-3.5]),
        }
        matrices = build_candidate_matrices(archived_decomposition(), baseline, raw_candidates)
        # The original Blend50 is -3. All original tree residuals stay unchanged
        # when replacing the Dense GLM, even though its new score is -1.
        np.testing.assert_array_equal(matrices["hierarchical_amount"], [[-4, -1, 2, 1, 0, -4, -2]])
        np.testing.assert_array_equal(matrices["residual_te"], [[-4, -2, 2, 1.5, 0, -4, -2]])
        np.testing.assert_array_equal(matrices["aligned_offset"], [[-4, -2, 2, -0.5, 0, -4, -2]])


if __name__ == "__main__":
    unittest.main()
