import unittest

from tokenizecode import CodeTokenizer, CodeParser


class TestKotlinParser(unittest.TestCase):
    def setUp(self):
        code_file = "/Users/johannes/projects/codebuddy/buddy-intellij-plugin/src/main/kotlin/com/codebuddy/intellijplugin/ui/ToolWindowActionsPanel.kt"

        with open(code_file, "rt") as f:
            code = f.read()

        self.code = code

    def test_parse(self):
        parser = CodeParser()

        out = parser.parse(self.code, "kotlin")

        out.tree.pprint()
        print(set(out.tree.nonterminals()))

        nonterminals_file = (
            "/Users/johannes/projects/codebuddy/multicoder/tokenizer/nonterminals.json"
        )

        import json

        with open(nonterminals_file, "rt") as f:
            nonterminals = json.load(f)

        # show which nonterminals are not in the nonterminals.json file
        print("unknown nonterminals")
        print(set(out.tree.nonterminals()) - set(nonterminals))
