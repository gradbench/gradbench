import torch


def least_squares(*, x, y):
    x = torch.cat((torch.ones(len(x), 1), torch.tensor(x)), dim=1)
    y = torch.tensor(y)

    def f(beta):
        epsilon = y - x @ beta
        return epsilon @ epsilon

    return f


def linear_regression(*, x, y, eta):
    f = least_squares(x=x, y=y)
    b = [0.0] + [0.0] * len(x[0])
    while True:
        beta = torch.tensor(b, requires_grad=True)
        loss = f(beta)
        loss.backward()
        b1 = (beta - eta * beta.grad).tolist()
        if b1 == b:
            break
        b = b1
    return b


def main():
    beta = linear_regression(
        eta=1e-4,
        x=[[10], [8], [13], [9], [11], [14], [6], [4], [12], [7], [5]],
        y=[8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    )
    print(beta)


if __name__ == "__main__":
    main()
