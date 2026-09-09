import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.cases import TEST_CASES
from evals.run import run_evaluation, make_config

class DummyArgs:
    def __init__(self, tests):
        self.setup = "all"
        self.tests = ",".join(tests)
        self.runs = 1
        self.workers = 1
        self.list_tests = False

def main():
    state_change_ids = [
        tc["id"] for tc in TEST_CASES if tc.get("category") == "state_change"
    ]
    args = DummyArgs(state_change_ids)
    config = make_config(args)
    run_evaluation(config, ["mcp", "no_tools", "code_interpreter"], state_change_ids, 1, 1)


if __name__ == "__main__":
    main()
