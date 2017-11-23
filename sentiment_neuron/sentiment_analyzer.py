from sentiment_neuron.encoder import Model


class SentimentAnalyzer:
    def __init__(self):
        self.model = Model()

    def analyze(self, texts):
        if isinstance(texts, list):
            return list(self.model.transform(texts)[:, 2388])
        elif isinstance(texts, str):
            return self.model.transform([texts])[:, 2388][0]
        else:
            raise ValueError('Unexpected texts type: {}'.format(type(texts)))
