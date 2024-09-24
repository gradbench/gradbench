from dataclasses import dataclass


@dataclass
class Functions:
    original: callable
    prepare: callable
    unwrap: callable

    def __call__(self, *args, **kwargs):
        return self.original(*args, **kwargs)


def wrap(wrap_in, wrap_out):

    def main_decorator(func):
        return Functions(original=func, prepare=wrap_in, unwrap=wrap_out)

    return main_decorator
