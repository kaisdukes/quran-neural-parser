import unittest

from src.parser.elas import Elas
from src.container import Container
from src.parser.oracle import Oracle


class OracleTest(unittest.TestCase):

    def setUp(self):
        self.container = Container()
        self.elas = Elas()

    def test_oracle(self):
        for expected_graph in self.container.syntax_service.graphs:

            output_graph = expected_graph.only_tokens()
            oracle = Oracle(expected_graph, output_graph)
            oracle.expected_actions()

            self.elas.compare(expected_graph, output_graph)

            # all output edges should be expected
            for output_edge in output_graph.edges:
                self.assertEqual(expected_graph.contains_edge(output_edge), True)

        self.assertEqual(self.elas.precision, 1.0)
        self.assertEqual(self.elas.recall, 0.945793687759221)


if __name__ == '__main__':
    unittest.main()
