import configparser
from unittest import TestCase
from demandprediction.utils.grid import PredictionArea


class TestGrid(TestCase):

    def test_createGrid(self):
        grid = PredictionArea.from_config('myconfig.ini')

        pass


    def test_with_border(self):
        self.fail()
