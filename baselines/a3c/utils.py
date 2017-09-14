import os


def make_path(f):
    return os.makedirs(f, exist_ok=True)


def constant(p):
    return 1


def linear(p):
    return 1-p

schedules = {
    'linear':linear,
    'constant':constant
}


class Scheduler(object):

    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.num_values = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v*self.schedule(self.n / self.num_values)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v*self.schedule(steps / self.num_values)
