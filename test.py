import torch


def res(num1, num2):
    if num1 == 1 and num2 == 1:
        return 10
    if num1 == 1 and num2 == 0:
        return 20
    if num1 == 0 and num2 == 1:
        return 30
    if num1 == 0 and num2 == 0:
        return 40



a = torch.tensor([0, 1, 2, 3, 4, 6, 7, 8])
b = torch.tensor([0, 1, 2, 3, 4, 6, 7, 8])
c = res(a, b)
print(a)
print(b)
print(c)







"""
a = torch.arange(10).reshape(5, 2)
print(a)
a = torch.split(a, [3, 2])
print(a)

"""