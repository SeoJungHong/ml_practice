# O
def f1(values: list[int]):
    return sum(x ** 2 for x in values if x % 2 == 0)


# X
def f2(values: list[int]):
    return sum(x ^ 2 for i, x in enumerate(values) if i % 2 == 0)


# O
def f3(values: list[int]):
    return sum(x * x if x % 2 == 0 else 0 for x in values)


# X
def f4(values: list[int]):
    return sum(x ^ 2 for x in values[::2])


# O
def f5(values: list[int]):
    return sum(x ** 2 if x % 2 == 0 else 0 for x in values)


# expected : 20
numbers = [1, 2, 3, 4, 5]

print('-- f1')
print(f1(numbers))
print('-- f2')
print(f2(numbers))
print('-- f3')
print(f3(numbers))
print('-- f4')
print(f4(numbers))
print('-- f5')
print(f5(numbers))
