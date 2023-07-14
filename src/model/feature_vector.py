from typing import List
from enum import Enum

from ..morphology.segment_type import SegmentType
from ..morphology.part_of_speech import PartOfSpeech
from ..morphology.mood_type import MoodType
from ..morphology.voice_type import VoiceType
from ..morphology.case_type import CaseType
from ..morphology.state_type import StateType
from ..morphology.pronoun_type import PronounType
from ..morphology.special_type import SpecialType
from ..lexicography.lemma_service import LemmaService
from ..syntax.syntax_node import SyntaxNode
from ..syntax.syntax_graph import SyntaxGraph
from ..syntax.relation import Relation
from ..syntax.phrase_type import PhraseType
from ..syntax.subgraph import subgraph_end
from ..parser.stack import Stack
from ..parser.queue import Queue


class FeatureVector:

    def __init__(self):
        self.values: List[float] = []

    def add_bit(self, bit: bool):
        self.values.append(1 if bit else 0)

    def add_enum(self, value: Enum | None, enum_type: type[Enum]):
        v = [0]*len(enum_type)
        if value is not None:
            v[value.value - 1] = 1
        self.values.extend(v)

    def add_tagged_enum(self, value: Enum | None, enum_type: type[Enum]):
        v = [0]*len(enum_type)
        if value is not None:
            v[value.value[0] - 1] = 1
        self.values.extend(v)

    def concat(self, v: List[float]):
        self.values.extend(v)


def get_feature_vector_and_lemma_ids(
        lemma_service: LemmaService,
        graph: SyntaxGraph,
        stack: Stack,
        queue: Queue):

    feature_vector = FeatureVector()
    lemma_ids: List[int] = []
    for x in [stack.node(0), stack.node(1), stack.node(2), queue.peek()]:

        feature_vector.add_tagged_enum(x.part_of_speech if x else None, PartOfSpeech)
        feature_vector.add_tagged_enum(x.phrase_type if x else None, PhraseType)

        s = x.segment if x else None
        feature_vector.add_enum(s.voice if s else None, VoiceType)
        feature_vector.add_tagged_enum(s.mood if s else None, MoodType)
        feature_vector.add_enum(s.case if s else None, CaseType)
        feature_vector.add_enum(s.state if s else None, StateType)
        feature_vector.add_enum(s.pronoun_type if s else None, PronounType)
        feature_vector.add_enum(s.type if s else None, SegmentType)
        feature_vector.add_tagged_enum(s.special if s else None, SpecialType)

        lemma = s.lemma if s is not None else None
        lemma_id = lemma_service.value_of(lemma) if lemma else lemma_service.count
        lemma_ids.append(lemma_id)

        for relation in Relation:
            feature_vector.add_bit(has_dependent(graph, x, relation))

        feature_vector.add_bit(is_valid_subgraph(graph, x))
        feature_vector.add_bit(is_edge(graph, stack))

    return feature_vector, lemma_ids


def has_dependent(graph: SyntaxGraph, head: SyntaxNode, relation: Relation):
    for edge in graph.edges:
        if edge.head is head and edge.relation is relation:
            return True
    return False


def is_valid_subgraph(graph: SyntaxGraph, node: SyntaxNode):
    if node is not None and not node.is_phrase and graph.head(node) is None:
        end = subgraph_end(graph, node)
        return end is not None and graph.head(end) is not None and graph.phrase(node, end) is None
    return False


def is_edge(graph: SyntaxGraph, stack: Stack):
    return (stack.node(0) is not None
            and stack.node(1) is not None
            and graph.edge(stack.node(0), stack.node(1)) is not None)
