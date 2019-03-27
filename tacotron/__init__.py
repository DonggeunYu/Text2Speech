from .tacotron import Tacotron

def create_model(hparams):
    return Tacotron(hparams)