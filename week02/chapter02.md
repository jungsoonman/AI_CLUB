
Week 2 
===
- - - 
## 데이터 분석을 위한 라이브러리
---
* #### 라이브러리
    * #####  Numpy(넘파이) : 기본적인 배열 처리나 수치 계산을 하는 라이브러리
    * #####  Scipy(사이파이) : 과학기술계산용 함수를 사용가능한 라이브러리
    * #####  pandas(판다스)  : 행과 열로 구성된 2차원 테이블 형태의 데이터를 조작 가공 분석하기 위한 라이브러리
    * #####  Matploib(매트플롯릿) : 그래프 제작을 위한 라이브러리 
    * #####  라이브러리 임포트
        *   (1). import 모듈 이름 as 식별이름
        *   (2). from   모듈 이름 import 속성
        *   ex) .```import numpy as np```
--- 
* #### 매직 명령어
    * #####  %precision 
        *   넘파이 기능 설정 : 소수 몇 번째 자리까지 표시할지 지정
            * ex).```%precision 3```
    * #####  %matplotlib
        *   매트플롯립 기능 설정 : 그래프 표시 방법을 지정
            * ex).```%matplotolib inline```
---
* #### 넘파이 기초 <br/>
    * #####  배열
        * 배열생성
            ```data = np.array([9 , 2, 3, 4, 10, 6, 7, ... 1, 5])```<br/><br/>
    * #####  데이터 형
        *   비트 수가 클수록 더 넓은 범위의 값을 표현할 수 있지만 더 많은 메모리가 필요
            ``` data.dtype ```
            ```출력값[dtype('int32')]```

* #### 차원과 원소 수<br/>
    * #####  차원
        * ```print('차원수: ', data.ndim)```
        * ```출력값[차원수 : 1]```
    * #####  원소수
        * ```print('원소수:', data.size)```
        * ```출력값[원소수:10]```
* #### 모든 원소에 대한 계산<br/>
    * #####  각 원소 간의 연산
        *   곱셈
            ```np.array([1 , 2, 3, 4, 5, 6, 7, 8, 9 ,10]) * np.array([10, 9, 8 , 7 , 6, 5 ,4 ,3 ,2 ,1])```
            ```결과 : [10 , 18 , 24, 28 ... ,24 ,18 ,10]```
        *   거듭제곱
            ```np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) ** 2```
            ```결과 : [1 4 9 16 ... 81 100]```
        *   나눗셈
            ```np.array([1 , 2, 3, 4, 5, 6, 7, 8, 9 ,10]) / np.array([10, 9, 8 , 7 , 6, 5 ,4 ,3 ,2 ,1])```
            ```결과 : [0.1 , 0.222, 0.375, 0.571 ... ,2.667 ,4.5 ,10.]```

* #### 정렬(sort) <br/>
    * #####   데이터를 정렬하려면 sort메서드를 사용
        *   sort
        ``` 
            print(data)  출력 [8 2 3 4 10 ... 1 5]
            
            data.sort() 
            print(data)  출력 [1 2 3 4 ... 8 9 10]

            data[::-1].sort()
            print(data)  출력 [10 9 8 7 6 ... 2 1]
        ```
* #### 최솟값 , 최댓값 합계 , 누적합 계산산
    * ##### 함수
        *   최솟값
            ```print(data.min())```
            ```출력[ 1 ]```
        *   최댓값
            ```print(data.max())```
            ```출력[ 10 ]```
        *   합계
            ```print(data.sum())```
            ```출력[ 55 ]```
        *   누적 합계
            ```print(data.cumsum())```
            ```출력[10 18 27 34 ... 54 55]```
        *   누적 비율
            ```print(data.cumsum()/ data.sum())```
            ```출력[0.182 0.345 0.491 ... 0.982 1]```
*   #### 난수 
    * ##### ramdom  라이브러리 사용
        * 임포트 
            ```import numpy.random as random```
        *   난수 시드
            ```random.seed(0)```
            > ##### 반드시 호출할 필요는 없지만 동일한 시드값을 지정함녀 여러변 난수를 방생시켜도 같은 난수를 얻을 수 있다.
        *   난수 배열 생성
            ```random.seed(0)```
            ```rnd_data = random.randn(10)```
            ```print(rnd_data)```
            ```출력[1.764 0.4 ... -0.141 0.411]```
            | **기능** | **의미**|
            |:-|:-|
            | rand | 균등분포. 0.0 이상 1.0 미만|
            |random_sample|균등분포 0.0 이상 1.0 미만 (rand와 인수 지정 방법이 다름)
            |randint|균등분포. 임의의 범위의 정수|
            |randn|정규분포. 평균 0 ,표준편차 1을 따르는 난수|
            |normal|정규분포. 임의의 평균과 표준편차를 따르는 난수|
            |binomial|이항분포를 따르는 난수|
            |beta|베타분포를 따르는 난수|
            |gamma|감마분포를 따르는 난수|
            |chisquare|카이제곱을 따르는 난수|
<br/>

*   ####    행렬
    *   #####   넘파이를 이용한 행렬 계산도 가능하다.
        ```
        array1 = np.arange(9).reshape(3,3)
        print(array1)
        출력 [[0 1 2]
              [3 4 5]
              [6 7 8]]
        첫번째 행
        array1[0,:]

        첫번째 열
        array1[:,0]
        ```
    *   #####   행렬연산
        *   ######  행렬의 곱
            ```
            # 3x3 행렬 생성
            array2 = np.arange(9,18).reshape(3,3)
            print(array2)

            #행렬 곱셈
            np.dot(array1 , array2)

            #각 원소의 곱
            array1 * array2

            #원소가 0 or 1인 행렬생성
            np.zeros(2,3)
            np.ones(2,3)
            ```
    ---
    *   ####    문제풀이
    ```
    import numpy as np
    import numpy.random as random

    # 1부터 50까지 자연수의 합을 계산하고 출력하는 프로그램
    def calcer():
        num_list = np.arange(1, 51)
        print(num_list)  # 배열 출력
        print(sum(num_list))  # 합 출력 (이 부분만 np.sum으로 바꾸면 더 일관성 있어요)

    calcer()

    # 표준 정규 분포를 따르는 10개의 난수 생성
    ran_num = random.randn(10)
    print(ran_num)
    print(min(ran_num))
    print(max(ran_num))
    print(sum(ran_num))

    #모든 원소가 3인 5 x 5 행렬을 만들고 이행렬의 제곱을 계산하세요.
    matrix = [[3] * 5 for _ in range(5)]
    print(matrix)
    print(np.dot(matrix, matrix))
    ```
    ---

*   ####    사이파이 기초 <br/>
    *   #####   임포트
        ```
        #선형대수 라이브러리
        import scipy.linalg as linalg

        #최적화 계산(최솟값)용 함수
        from scipy.optimize import minimize_scalar
        ```
    

    *   #####   행렬식과 역행렬 계산
        + ######   det 함수를 사용
        ```
        matrix = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])

        #역행렬
        print(linalg.det(matrix))
        ```
    *   #####   고유값과 고유벡터
        + ######  eig 함수를 사용
        ```
        #고윳값과 고유벡터
        eig_value, eig_vector = linalg.eig(matrix)
        ```
    *   #####   뉴턴법
        +   $f(x) = x^2 + 2x +1$
        ```
        #함수 정의
        def my_function(x):
            return (x**2 + 2*2 +1)
        
        #뉴턴법 임포트
        from scipy.optimize import  newton
        #연산 실행
        newton(my_function,0)

        #최솟값 구하기
        minimize_scalar(my_function, method = 'Brent')
        ```
    *   ##### 연습문제
    ```
    import numpy as np
    import numpy.random as random
    import scipy as sc
    from scipy.optimize import newton
    from scipy.optimize import minimize_scalar
    import scipy.linalg as linalg

    #my_function 함수의 계산식을 f(x) = 0 부터 다양한 함수로 변경해 최솟값 등을 계산해 봅시다.
    def my_function(x):
        return np.exp(2*x)+1

    x = 0
    # print(my_function(x))
    # print(minimize_scalar(my_function , method ='Brent'))

    #다음 행렬에 대해 행력식을 구하시오
    A = np.array([[1,2,3],[1,3,2],[3,1,2]])
    # print(A)

    # print(np.linalg.eig(A))

    def my_function(x):
        return x**3 + 2*x + 1

    print()
    print(my_function(newton(my_function,0)))
    ```
    ---
    *   ####    판단스 기초
        *  #####    라이브러리 import
        ```        from pandas import Series, DataFrame        ```
        *   #####   Series 사용법
            - ######   Series는 1차원 배열이며, 타입은 데이터형형
            ```
            #Series
            sample_pandas_data = 
            #인덱스 생성
            sample_pandas_index_data = pd.Series([0,10,20,30,40,50,60,70,80,90],index=['a','b','c'...'h','i','j'])
            ```
        *   #####    DataFrame 사용법
            - ######    DataFrame 객체는 2차원 배열이다.
            ```
            #딕셔너리 생성
            dictionary={
                'ID':[100,101...104],
                'City':['Seoul','Pusan'...'Gangneung','Seoul'],
                'Birth_year':[1990,1989,1992...1982],
                'Name':['Junho','Heejin',...'Steve']
            }
            #데이터 프레임 생성
            data_frame = DataFrame(dictionary)

            #인덱스 생성
            data_frame = DataFrame(dictionary, index=['a','b','c','d','e'])
            ```
        *   #####   행렬 다루기
            -   특정열 추출
            ```data_frame.Birth_year```
            -   데이터 추출(필터)
            ```data_frame[data_frame['City']=='Seoul']```
            -   다중 (필터)
            ```data_frame[data_frame['City'].isin(['Seoul','Pusan'])]```
            -   열과 행삭제 (행 : 0 , 열 : 1)
            ```data_frame.drop(['Birth_year'] , axis =1 )```
            -   데이터 결합
            ```pd.merge(데이터1 ,데이터2)```
            -   집계 (성별간 수학 평균)
            ```data_frame.groupby('Sex')['Math'].mean()```
            -   정렬 및 null
                ```
                #인덱스 순으로 정렬.
                data_frame.sort_index()
                #값을 기준으로 정렬
                data_frame.Birth_year.sort_values()
                #null값 판정 (null 인 값은 true 반환)
                data_frame.isnull()
                ```

        *   #####   연습문제
            ```
            from pandas import Series,DataFrame
            import pandas as pd

            data1={
                'ID':['1','2','3','4','5'],
                'Sex':['F','F','M','M','F'],
                'Money':[1000,2000,500,300,700],
                'Name':['Suji','Minji','Taeho','Jinsung','Suyoung']
            }

            data1_frame = DataFrame(data1)
            print(data1_frame[data1_frame['Money']>=500])

            #데이터에서 성별 평균 Money
            print(data1_frame.groupby('Sex')['Money'].mean())

            data2= {'ID': ['3','4','7']
                    ,'Math':[60,30,40]
                    ,'English':[80,20,30]}
            data2_frame = DataFrame(data2)

            merge_data = pd.merge(data1_frame,data2_frame)
            print(merge_data['Math'].mean() , merge_data['English'].mean())
            ```
    <br/>

    ---
    *   ####    매트플롯립 기초
        -   #####   import 라이브러리
            ```
            #매트플롯립과 씨본 불러오기
            import matplotlib as mpl
            import seaborn as sns

            #pyplot 은 plt이라는 이름으로 실행 가능
            import matplotlib.pyplot as plt

            #주피터 노트북에서 그래프를 표시하기 위해 필요한 매직 명령어
            %matplotlib inline
            ```
        -   #####   산점도
            +   ######  점으로 표시하는 그래프
            ```
                #산점도

                #x 축 데이터
                x = np.random.radn(30)
                
                #y 축 데이터
                y = np.sin(x) + np.random.randn(30)

                #그래프 크기 설정
                plt.figure(figsize=(20,6))

                #산점도 생성 방법1 
                plt.plot(x, y 'o')

                #타이틀
                plt.title('Title Name')
                #X좌표 이름
                plt.xlabel('X')
                #Y좌표 이름
                plt.ylabel('Y')

                #grid 표시
                plt.grid(True)
            ```
        +   ######  연속형 그래프
            ```
                #연속형 곡선
                
                #데이터 범위
                numpy_data_x = np.arange(1000)
                
                #난수 발생과 누적 합계
                numpy_random_data_y = np.random.randn(1000).cumsum()

                #그래프 크기 지정
                plt.figure(figsize=(20,6))

                #label = 과 legend로 레이블 표시할 수 있다.
                plt.plot(numpy_data_x ,numpy_random_data_y ,label= 'Label')

                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
            ```
        +   ######  그래프 분할
            ```
            #2행 1열 그래프 생성

            #그래프 크기설정
            plt.figure(figsize=(20,6))
            #2행 1열 그래프의 첫번째
            plt.subplot(2,1,1)

            x = np.linspace(-10,10,100)
            plt.plot(x, np.sin(x))

            #2행 1열 그래프 생성
            plt.subplot(2,1,2)
            y= np.linspace(-10,10,100)
            plt.plot(y,np.sin(2*y))

            plt.grid(True)
            ```
        +   ######  함수그래프 그리기
            $f(x) = x^2 + 2x + 1$
            ```
            #함수정의

            def my_function(x):
                return x**2 + 2*x + 1

            x = np.arange(-10,10)
            plt.figure(figsize=(20,6))
            plt.plot(x, my_function(x))
            plt.grid(True)
            ```
        +   ######  히스토그램 (hist 메서드 사용)
            ```
            #그래프 크기 지정
            plt.figure(figsize =(20,6))
            
            #히스토그램 생성
            plt.hist(np.random.radn(10**5)*10 +50 , bins = 60, range = (20,80))

            plt.grid(True)
            ```
        +   ##### 연습문제
        ```
        import matplotlib.pyplot as plt
        import matplotlib as mlp
        import seaborn as sns
        ```

         $ y = 5x + 3 (x 는 -10 부터 10까지) $
        ```
        def function(x):
            return 5*x + 3
        x = np.linspace(-10,10,100)

        plt.plot(x,function(x))

        x1 = np.linspace(-10,10,100)
        plt.plot(x1, np.sin(x1), x1, np.cos(x1))

        x2 = np.random.uniform(0.0, 1.0,1000)
        plt.hist(x2, bins= 60, range=(0.0,1.0))
        plt.grid(True)
        plt.show()
        ```



   
    
        
        






