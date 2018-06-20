import attr
import chess
import math

from .constants import (
    TC_MOVETIME,
    TC_WTIME,
    TC_BTIME,
    TC_WINC,
    TC_BINC,
    TC_MOVESTOGO,
    TC_KEYS,
    TC_SUDDEN_DEATH_THRESHOLD,
    TC_OPPONENT_TIME_RATIO,
)


@attr.s
class TimeManager():
    total_time = attr.ib(default=None)
    total_moves = attr.ib(default=None)

    def handle_movetime(self, data):
        return data[TC_MOVETIME]

    def handle_fischer(self, color, data):
        if color == chess.WHITE:
            time = data[TC_WTIME]
            otime = data[TC_BTIME]
            inc = data[TC_WINC]
        else:
            time = data[TC_BTIME]
            otime = data[TC_WTIME]
            inc = data[TC_BINC]

        if time < TC_SUDDEN_DEATH_THRESHOLD:
            # so little time, panic!
            return self.handle_sudden_death(color, data)

        ratio = max(otime/time, 1.0)
        # assume we have 16 moves to go
        moves = 16 * min(2.0, ratio)
        return time / moves + 3 / 4 * inc

    def handle_classic(self, color, data):
        if self.total_time is None and self.total_moves is None:
            # first time getting time control information
            # assume this is the start
            self.total_moves = data[TC_MOVESTOGO]
            if color == chess.WHITE:
                self.total_time = data[TC_WTIME]
            else:
                self.total_time = data[TC_BTIME]
        if color == chess.WHITE:
            time = data[TC_WTIME]
            otime = data[TC_BTIME]
        else:
            time = data[TC_BTIME]
            otime = data[TC_WTIME]
        if time < TC_SUDDEN_DEATH_THRESHOLD:
            # so little time, panic!
            return self.handle_sudden_death(color, data)
        moves = data[TC_MOVESTOGO]
        tc = time / moves
        if self.total_moves:
            tc_cf = time + self.total_time
            tc_cf /= moves + self.total_moves
        else:
            tc_cf = math.inf
        time_to_spend = min(tc, tc_cf)
        if time / otime < TC_OPPONENT_TIME_RATIO:
            # we have a lot more time than our opponent
            # so let's spend less time on the next move
            time_to_spend *= TC_OPPONENT_TIME_RATIO
        return time_to_spend

    def handle_sudden_death(self, color, data):
        if color == chess.WHITE:
            time = data[TC_WTIME]
            inc = data.get(TC_WINC, 0)
        else:
            time = data[TC_BTIME]
            inc = data.get(TC_BINC, 0)

        # assume we'll play 20 more moves
        time_to_spend = time / 20

        # total panic mode!
        if time < 500:
            return 100
        elif time < 1000:
            return min(time_to_spend, inc / 2)
        elif time < 2000:
            return min(time_to_spend, inc)

        return time_to_spend

    def handle(self, color, args):
        data = parse_time_control(args)
        return self.calculate_duration(color, data)

    def calculate_duration(self, color, data):
        if TC_MOVETIME in data:
            duration = self.handle_movetime(data)
        elif TC_MOVESTOGO in data:
            duration = self.handle_classic(color, data)
        elif TC_WINC in data and TC_BINC in data:
            duration = self.handle_fischer(color, data)
        else:
            duration = self.handle_sudden_death(color, data)
        return max(duration, 10) / 1000


def parse_time_control(args):
    data = {}
    args = args.split()
    for i in range(len(args)):
        token = args[i]
        if token in TC_KEYS:
            data[token] = float(args[i+1])
    return data
