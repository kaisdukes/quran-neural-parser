from typing import List

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from .multilayer_perceptron import MultilayerPerceptron
from .parser_dataset import ParserDataset
from .model import Model
from ..morphology.part_of_speech import PartOfSpeech
from ..lexicography.lemma_service import LemmaService
from ..syntax.syntax_graph import SyntaxGraph
from ..parser.parser import Parser
from ..parser.parser_action import ParserAction, encode_parser_action
from ..parser.action_type import ActionType
from ..parser.elas import Elas


def train_and_test(
        lemma_service: LemmaService,
        train_graphs: List[SyntaxGraph],
        test_graphs: List[SyntaxGraph]):

    print('Preparing data...')
    training_data = ParserDataset(lemma_service, train_graphs)
    print(f'{len(training_data._feature_vectors)} training examples')

    num_tokens_queue_stack = 4
    embedding_dim = 10
    input_dim = len(training_data._feature_vectors[0]) + num_tokens_queue_stack * embedding_dim
    hidden_dim = 200
    output_dim = encode_parser_action(ParserAction(ActionType.EMPTY, PartOfSpeech.VERB)) + 1
    num_lemmas = lemma_service.count
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    f1_threshold = 0.86

    data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    multilayer_perceptron = MultilayerPerceptron(input_dim, hidden_dim, output_dim, num_lemmas, embedding_dim)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(multilayer_perceptron.parameters(), lr=learning_rate)

    # train
    for epoch in range(num_epochs):
        for feature_vector, lemma_ids, label in data_loader:

            # forward pass
            outputs = multilayer_perceptron(feature_vector, lemma_ids)
            loss = loss_function(outputs, label)

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        # test
        elas = Elas()
        model = Model(multilayer_perceptron)
        for expected_graph in test_graphs:
            output_graph = expected_graph.only_tokens()
            parser = Parser(model, lemma_service, output_graph)
            try:
                parser.parse()
            except Exception as e:
                print(e)
            elas.compare(expected_graph, output_graph)

        print(f'F1 score: {elas.f1_score}')
        if elas.f1_score >= f1_threshold:
            break
