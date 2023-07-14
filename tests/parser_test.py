import unittest

from split_treebank import split_treebank
from src.container import Container
from src.model.train import train_and_test


class ParserTest(unittest.TestCase):

    def setUp(self):
        self.container = Container()

    def test_ten_fold_cross_validation(self):
        for fold in range(10):
            self._train_and_test(fold)

    def _train_and_test(self, fold: int):
        print(f'Fold {fold}')
        (train_graphs, test_graphs) = split_treebank(self.container.syntax_service, fold)
        train_and_test(
            self.container.lemma_service,
            train_graphs,
            test_graphs)


if __name__ == '__main__':
    unittest.main()
