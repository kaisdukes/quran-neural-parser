from typing import List

import torch
from torch.utils.data import Dataset

from .feature_vector import get_feature_vector_and_lemma_ids
from ..lexicography.lemma_service import LemmaService
from ..parser.parser import Parser
from ..parser.oracle import Oracle
from ..parser.stack import Stack
from ..parser.queue import Queue
from ..parser.parser_action import ParserAction, encode_parser_action
from ..syntax.syntax_graph import SyntaxGraph


class ParserDataset(Dataset):

    def __init__(self, lemma_service: LemmaService, graphs: List[SyntaxGraph]):
        self._feature_vectors: List[List[float]] = []
        self._lemma_ids: List[List[int]] = []
        self._labels: List[int] = []

        # parse each graph
        for expected_graph in graphs:

            # apply expected actions
            actions = Oracle(expected_graph, expected_graph.only_tokens()).expected_actions()
            output_graph = expected_graph.only_tokens()
            parser = Parser(None, None, output_graph)
            stack = parser.stack
            queue = parser.queue
            for action in actions:
                self._add_instance(lemma_service, output_graph, stack, queue, action)
                parser.execute(action)

            # stop parsing
            self._add_instance(lemma_service, output_graph, stack, queue, None)

    def __len__(self):
        return len(self._feature_vectors)

    def __getitem__(self, index):
        return (
            torch.tensor(self._feature_vectors[index], dtype=torch.float32),
            torch.tensor(self._lemma_ids[index], dtype=torch.long),
            torch.tensor(self._labels[index], dtype=torch.long)
        )

    def _add_instance(self,
                      lemma_service: LemmaService,
                      graph: SyntaxGraph,
                      stack: Stack,
                      queue: Queue,
                      action: ParserAction | None):

        feature_vector, lemma_ids = get_feature_vector_and_lemma_ids(
            lemma_service,
            graph,
            stack,
            queue)

        self._feature_vectors.append(feature_vector.values)
        self._lemma_ids.append(lemma_ids)
        self._labels.append(encode_parser_action(action))
