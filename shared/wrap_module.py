def wrap(wrap_in, wrap_out):
    def main_decorator(func):
        def args(input):
            inputs = wrap_in(input)
            ret = func(inputs)
            return wrap_out(ret)

        return args

    return main_decorator
