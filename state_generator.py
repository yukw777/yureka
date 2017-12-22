import attr
from chess import pgn


@attr.s
class StateGenerator():
    game_file_name = attr.ib()

    def __attrs_post_init__(self):
        self.game_file = open(self.game_file_name, 'r')

    def get_game(self):
        while True:
            g = pgn.read_game(self.game_file)
            if g is None:
                break
            yield g

    def generate(self):
        for game in self.get_game():
            pass
