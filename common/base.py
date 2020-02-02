from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(self, env):
        self._env = env

    @property
    def env(self):
        return self._env

    @abstractmethod
    def choose_action(self):
        ''' Abstact method to be replaced by the policy prediction '''
        raise NotImplementedError

    @abstractmethod
    def perform_train(self):
        ''' Abstact method to be replaced by the training loop '''
        raise NotImplementedError

    @abstractmethod
    def perform_test(self):
        ''' Abstact method to be replaced by the test loop '''
        raise NotImplementedError
