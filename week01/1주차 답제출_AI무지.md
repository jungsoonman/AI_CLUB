# 1주차문제풀이_AI무지

## 👤 김성민 문제

# 평가 문제 1

## 문제:

숫자 출력 형식을 지정하는 방법을 이용해, 원주율의 값을 소수점 이하 6자리까지 출력하는 코드를 입력하세요. (입력 종료는 엔터 두 번)

### 진행 목적:

- input을 다음과 같이 받아서 정확한 출력이 되는지 판단
- 원주율 (math.pi)을 소수점 6자리로 포맷

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

print("문제 : 숫자 출력 형식을 지정하는 방법을 이용해, 원주율의 값을 소수점 이하 6자리까지 출력하는 코드를 입력하세요. 
(입력 종료는 엔터 두 번)")

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

## 제출한 정답:

```python
print("{:0.6f}".format(math.pi))
```

---

# 문제: a번부터 b번사이의 피보나치 수의 합 구하기

## 제약:

- 첫 줄에 두 수 a, b를 공백으로 구분하여 입력.
- a번부터 b번사이의 피보나치 수의 합을 출력.
- (조건: 0 < a < b)

## 피보나치란?

- 카운터는 1, 1, 2, 3, 5, 8, 13, ... 같이, 각 타령 수는 전 두 타령의 합입니다.

## 제출한 정답:

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

## 테스트 예시:

### Test Case 1:

```text
Input : 1 2
Output : 1
```

### Test Case 2:

```text
Input : 2 3
Output : 2
```

### Test Case 3:

```text
Input : 11 63
Output : (해당 구간의 피보나치 수의 합 출력)
```

---

## 👤 양승민 문제


# 🧮 문제 1: 리스트 조작하기

1부터 20까지의 숫자 중에서 **짝수만 포함하는 리스트**를 만들고, 각 숫자를 **3배로 늘린 결과**를 출력하세요.  
(💡 리스트 컴프리헨션 및 Numpy 모두 활용)

---

### ✅ 파이썬 문법

```python
# 1) 1부터 20까지 숫자 생성
num_list = [i for i in range(1, 21)]

# 2) 짝수만 추출
even_num_list = [i for i in num_list if i % 2 == 0]

# 3) 각 숫자를 3배로 증가
trio_num_list = [i * 3 for i in even_num_list]

print("파이썬 문법) 짝수만 생성:", even_num_list)
print("파이썬 문법) 숫자 3배 생성:", trio_num_list)
```

---

### ✅ Numpy 문법

```python
import numpy as np

# 1) 1~20 배열 생성
np_num_list = np.arange(1, 21)

# 2) 짝수만 남기기
np_even_list = np_num_list[np_num_list % 2 == 0]

# 3) 각 숫자에 3을 곱함
trio_num_list = np_even_list * 3

# 4) 각 숫자를 3번씩 반복 (repeat)
repeat_num_list = np_even_list.repeat(3)

print("넘파이 문법) 짝수만 생성:", np_even_list)
print("넘파이 문법) 숫자 3배 생성:", trio_num_list)
print("넘파이 repeat() 사용:", repeat_num_list)
```

---

# 📚 문제 2: 딕셔너리 다루기

주어진 학생 점수 딕셔너리에서 다음을 수행하세요:

```python
scores = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'David': 95, 'Eva': 88}
```

---

## ✅ 1. 모든 학생 이름과 점수 출력

```python
dictionary = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'David': 95, 'Eva': 88}

for i in dictionary:
    print("{:<8} : {:>1}".format(i, dictionary[i]))
```

출력 예:
```
Alice    : 85
Bob      : 92
Charlie  : 78
David    : 95
Eva      : 88
```

---

## ✅ 2. 가장 높은 점수를 받은 학생 출력

```python
student = max(dictionary, key=dictionary.get)
print("가장 높은 점수 {:<2} 학생은 {:>1}".format(dictionary[student], student))
```

출력 예:
```
가장 높은 점수 95 학생은 David
```

---

## ✅ 3. 90점 이상인 학생 이름만 리스트로 출력

```python
students = []

for i in dictionary:
    if dictionary[i] >= 90:
        students.append(i)

print("이름     |  점수")
for i in students:
    print("{:<8} | {:>3}".format(i, dictionary[i]))
```

출력 예:
```
이름     |  점수
Bob      |  92
David    |  95
```

