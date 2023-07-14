import torch

from .multilayer_perceptron import MultilayerPerceptron
from .feature_vector import get_feature_vector_and_lemma_ids
from ..syntax.syntax_graph import SyntaxGraph
from ..parser.stack import Stack
from ..parser.queue import Queue
from ..parser.parser_action import decode_parser_action
from ..lexicography.lemma_service import LemmaService


class Model:
    def __init__(self, multilayer_perceptron: MultilayerPerceptron):
        self._multilayer_perceptron = multilayer_perceptron

    def action(
        self,
            lemma_service: LemmaService,
            graph: SyntaxGraph,
            stack: Stack,
            queue: Queue):

        feature_vector, lemma_ids = get_feature_vector_and_lemma_ids(
            lemma_service,
            graph,
            stack,
            queue)

        outputs = self._multilayer_perceptron(
            torch.tensor(feature_vector.values, dtype=torch.float32).unsqueeze(0),
            torch.tensor(lemma_ids, dtype=torch.long).unsqueeze(0),
        )

        action = torch.argmax(outputs).item()
        return decode_parser_action(action)
