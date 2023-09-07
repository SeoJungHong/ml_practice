import re
from collections.abc import Iterable

# 아래와 같은 로그 형식에서 시간만 추출하기
logs = ['2023-05-01 10:20:05 200 GET /page1.html',
        '2023-05-01 11:15:32 404 GET /page2.html']


def parse1(lines) -> Iterable[str]:
    for line in lines:
        yield line.split()[1]


print('-- parse1')
print(list(parse1(logs)))


def parse2(lines) -> Iterable[str]:
    for line in lines:
        yield from re.findall(r'\d{2}:\d{2}:\d{2}', line)[:1]


print('-- parse2')
print(list(parse2(logs)))


# def parse3(lines) -> Iterable[str]:
#     for line in lines:
#         yield re.findall(r'\d:\d:\d', line)[0]


# print('-- parse3')
# print(list(parse3(logs)))


def parse4(lines) -> Iterable[str]:
    for line in lines:
        yield from line.split(' ')[1]


print('-- parse4')
print(list(parse4(logs)))


def parse5(lines) -> Iterable[str]:
    for line in lines:
        yield re.findall(r'\d+.\d+.\d+', line)[0]


print('-- parse5')
print(list(parse5(logs)))
