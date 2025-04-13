
# 1주차문제풀이_AI무지

---

## 👤 김성민 문제

### 문제 1 - 숫자 출력 형식 지정 (원주율 소수점 6자리)

**문제 설명:**

숫자 출력 형식을 지정하는 방법을 이용해, 원주율의 값을 소수점 이하 6자리까지 출력하는 코드를 입력하세요. (입력 종료는 엔터 두 번)

```python
import math, io, sys

def main(input_code):
    origin_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        exec(input_code)
        output = sys.stdout.getvalue().strip()
        answer = format(math.pi, ".6f")
        if output == answer:
            return f"Correct! (Output: {output})"
        else:
            return f"Incorrect! (Output: {output})"

    except Exception as e:
        return f"An Error Occurred during executing code. : {e}"

    finally:
        sys.stdout = origin_stdout

print("문제 : 숫자 출력 형식을 지정하는 방법을 이용해, 원주율의 값을 소수점 이하 6자리까지 출력하는 코드를 입력하세요. (입력 종료는 엔터 두 번)")

lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

input_code = "
".join(lines)
result = main(input_code)
print(result)
```

**제출한 정답:**

```python
print("{:0.6f}".format(math.pi))
```

---

### 문제 2 - 피보나치 수의 합 구하기

**문제 설명:**

- 첫 줄에 두 수 a, b를 공백으로 구분하여 입력.
- a번부터 b번 사이의 피보나치 수의 합을 출력 (조건: 0 < a < b)

```python
def fibonacci(num_list):
    first, second = num_list.split()

    list = sorted([int(first), int(second)])

    sum = 0
    fibonacci_list = [i for i in range(1, list[1]+1)]

    for i in range(len(fibonacci_list)):
        if i == 0 or i == 1:
            fibonacci_list[i] = 1
        else:
            fibonacci_list[i] = fibonacci_list[i-2] + fibonacci_list[i-1]

        if i+1 >= list[0]:
            sum += fibonacci_list[i]

    print(sum)
```

**테스트 예시:**

- Input: 1 2 → Output: 1  
- Input: 2 3 → Output: 2  
- Input: 11 63 → Output: (해당 구간의 피보나치 수의 합 출력)

---

## 👤 양승민 문제

### 문제 1 - 리스트 조작하기 (Python / Numpy)

**문제 설명:**

1부터 20까지의 숫자 중에서 짝수만 포함하는 리스트를 만들고, 각 숫자를 3배로 늘린 결과를 출력하세요. (리스트 컴프리헨션 및 Numpy 모두 활용)

```python
# Python
num_list = [i for i in range(1, 21)]
even_num_list = [i for i in num_list if i % 2 == 0]
trio_num_list = [i * 3 for i in even_num_list]

print("파이썬 문법) 짝수만 생성:", even_num_list)
print("파이썬 문법) 숫자 3배 생성:", trio_num_list)
```

```python
# Numpy
import numpy as np
np_num_list = np.arange(1, 21)
np_even_list = np_num_list[np_num_list % 2 == 0]
trio_num_list = np_even_list * 3
repeat_num_list = np_even_list.repeat(3)

print("넘파이 문법) 짝수만 생성:", np_even_list)
print("넘파이 문법) 숫자 3배 생성:", trio_num_list)
print("넘파이 repeat() 사용:", repeat_num_list)
```

---

### 문제 2 - 딕셔너리 다루기

**문제 설명:**

주어진 학생 점수 딕셔너리에서 다음을 수행하세요:

1. 모든 학생 이름과 점수 출력
2. 가장 높은 점수를 받은 학생 출력
3. 90점 이상인 학생 이름만 리스트로 출력

```python
dictionary = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'David': 95, 'Eva': 88}

# 1. 이름과 점수 출력
for i in dictionary:
    print("{:<8} : {:>1}".format(i, dictionary[i]))

# 2. 최고 점수 학생
student = max(dictionary, key=dictionary.get)
print("가장 높은 점수 {:<2} 학생은 {:>1}".format(dictionary[student], student))

# 3. 90점 이상 학생
students = []
for i in dictionary:
    if dictionary[i] >= 90:
        students.append(i)

print("이름     |  점수")
for i in students:
    print("{:<8} | {:>3}".format(i, dictionary[i]))
```

---

## 👤 AI 무지 문제

### 문제 1 - 가위바위보 시뮬레이션

**문제 설명:**

A와 B는 총 3번의 가위바위보 게임을 진행합니다. 각 게임은 비길 경우 다시 진행하며, 승부가 나야 1게임으로 인정됩니다. 각 라운드의 승자와 비긴 횟수, 최종 승자, 그리고 최종 승자의 이력을 출력합니다. 가위바위보는 랜덤으로 결정합니다.

```python
import random

A_result = []
B_result = []

def RockScissorsPaper(A_result, B_result):
    RSP = {1: "가위", 2: "바위", 3: "보"}
    winner = []
    A_winner_count = 0
    B_winner_count = 0

    for i in range(1, 4):
        count = 0
        while True:
            A = random.randint(1, 3)
            B = random.randint(1, 3)

            if A == B:
                A_result.append(A)
                B_result.append(B)
                count += 1
                continue
            elif (A == 1 and B == 2) or (A == 2 and B == 3) or (A == 3 and B == 1):
                winner.append("B")
                B_winner_count += 1
                A_result.append(A)
                B_result.append(B)
                break
            else:
                winner.append("A")
                A_winner_count += 1
                A_result.append(A)
                B_result.append(B)
                break

        print("[", i, " 라운드 승자", winner[i - 1], ", 비긴횟수 :", count, "]")

    if A_winner_count > B_winner_count:
        print(["최종 승리자 A"])
        print("A 가 낸 가위바위보 이력:", [RSP[i] for i in A_result])
    else:
        print(["최종 승리자 B"])
        print("B 가 낸 가위바위보 이력:", [RSP[i] for i in B_result])

RockScissorsPaper(A_result, B_result)
```

---

### 문제 2 - 3x3 행렬 생성 및 분석

**문제 설명:**

각 원소의 값이 행 번호 + 열 번호인 3 x 3 행렬을 생성하세요. 아래 내용을 함께 출력하세요:

- 대각선 원소 배열 (왼쪽 위 → 오른쪽 아래)
- 대각선 원소의 합
- 가장 큰 합을 가진 행
- 가장 작은 합을 가진 열 (추가 구현 가능)

```python
matrix = [[i + 1, i + 2, i + 3] for i in range(1, 4)]
print("행렬:", matrix)

diag_list = [matrix[i][i] for i in range(len(matrix))]
print("대각원소:", diag_list)

diag_sum = sum(diag_list)
print("대각원소합:", diag_sum)

row_list = [sum(matrix[i]) for i in range(len(matrix))]
print("가장 큰 행:", matrix[row_list.index(max(row_list))])
```

---

### 문제 3 - 프로그래머스 1단계 문제 풀이

**문제 설명:**

프로그래머스 1단계 문제 중 하나를 선택하여 풀이하세요. (예: 선물 주고받기 문제)

```python
def solution(friends, gifts):
    n = len(friends)
    name_to_idx = {name: i for i, name in enumerate(friends)}

    give = [[0] * n for _ in range(n)]
    for gift in gifts:
        sender, receiver = gift.split()
        give[name_to_idx[sender]][name_to_idx[receiver]] += 1

    sent = [sum(give[i]) for i in range(n)]
    received = [sum(give[j][i] for j in range(n)) for i in range(n)]
    score = [sent[i] - received[i] for i in range(n)]

    next_gift = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if give[i][j] > give[j][i]:
                next_gift[i] += 1
            elif give[i][j] == give[j][i] and score[i] > score[j]:
                next_gift[i] += 1

    return max(next_gift)
```
