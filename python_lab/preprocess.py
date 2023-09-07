# X
def fill1(temperatures: list[float | None]) -> None:
    total, count = 0, 0
    for t in temperatures:
        if t is not None:
            total += t
            count += 1
    avg = total / count
    for t in temperatures:
        if t is None:
            t = avg


# X
def fill2(temperatures: list[float | None]) -> None:
    avg = sum(temperatures) / len(temperatures)
    for i in range(len(temperatures)):
        if temperatures[i] is None:
            temperatures[i] = avg


# O
def fill3(temperatures: list[float | None]) -> None:
    good = [t for t in temperatures if t]
    avg = sum(good) / len(good)
    for i, t in enumerate(temperatures):
        temperatures[i] = t or avg


# X
def fill4(temperatures: list[float | None]) -> None:
    good = [t for t in temperatures if t is not None]
    avg = sum(good) / len(good)
    temperatures = [t or avg for t in temperatures]


# O
def fill5(temperatures: list[float | None]) -> None:
    good = [t for t in temperatures if t is not None]
    avg = sum(good) / len(good)
    temperatures[:] = [t if t is not None else avg for t in temperatures]


temp = [0.3, 1.5, 8, 9.1, 0.7, None, 0.42, None, 99]

print('before')
print(temp)
print('after')
fill5(temp)
print(temp)
