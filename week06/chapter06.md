# Pandas 
--- 

* ### Importing libraries

    ```python
    #Please import the following libraries in davance, as they will be used later.
    import numpy as np
    import numpy.random as random
    import scipy as sp
    import pandas as pd
    from pandas import Series, DataFrame

    #Visualization library
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    %matplotlib inline

    #Round to three decimal places
    %precision 3 
    ```

* ### Hierarchical Index

    ```python
    hier_df = DataFrame(
        np.arange(9).reshape(3,3)),
        index = [
            ['a','a','b'],
            [1,2,2]
        ],
        columns = [
            ['Pusan','Seoul','Pusan'] , 
            ['Blue','Red','Red']
        ]

        #Result
        Pusan	Seoul	Pusan
        Blue	Red	Red
        a	1	0	1	2
        2	3	4	5
        b	2	6	7	8

    # naming the index
    hier_df.index.names = ['key1','key2']

    # naming the columns
    hier_df.columns.names=['city','color']

    #Reduce the column range
    hier_df['Pusan']

    #index-based aggregation
    hief_df.sum(level = 'key2', axis=0)

    #sum columns
    hier_df.sum(level = 'color',axis =1)

    #Drop an index value
    hier_df.drop(['b'])
    ```

*   ### Data merging

    ```python
    #Create data1
    data1 ={'id' : ['100','101','102','103','104','106','108','110','111','113'],
    'city' :['Seoul','Pusan','Daegu','Gangneung','Seoul','Seoul','Pusan','Daegu','Gangneung','Seoul'],
    'birth_year':[1990,1989,1992,1997,1982,1991,1988,1990,1995,1981],
    'name' :['Junho','Heejin','Mijung','Minho','Steeve','Mina','Sumi','Minsu','Jinhee','Daeho'],
    }

    df1 = DataFrame(data1)
    df1

    #Result
    	id	city	    birth_year	name
    0	100	Seoul	    1990	    Junho
    1	101	Pusan	    1989	    Heejin
    2	102	Daegu	    1992	    Mijung
    3	103	Gangneung	1997	    Minho
    4	104	Seoul	    1982	    Steeve
    5	106	Seoul	    1991	    Mina
    6	108	Pusan	    1988	    Sumi
    7	110	Daegu	    1990	    Minsu
    8	111	Gangneung	1995	    Jinhee
    9	113	Seoul	    1981	    Daeho

    #Create data2
    data2 ={'id' : ['100','101','102','105','107'],
    'math' : [90,80,70,60,50],
    'english' : [80,70,60,50,40],
    'sex' : ['M','F','F','M','M'],
    'index_num' : [0,1,2,3,4]
    }
    df2  = DataFrame(data2)
    df2 

    #Result
    	id	math	english	sex	index_num
    0	100	90	80	M	0
    1	101	80	70	F	1
    2	102	70	60	F	2
    3	105	60	50	M	3
    4	107	50	40	M	4
    ```

    -   #### merge strategy

        - ###### INNER JOIN
            The datasets are merged when both sides have a common key
        - ###### FULL JOIN
            All data from both datasets is merged 
        - ###### LEFT JOIN
            The merge is based on the key values of the left dataset
        - ###### RIGHT JOIN
            The merge is based on the key values of the right dataset

    
    -   #### INNER JOIN
        Use the "on" parameter to specify the key column for the merge

        ```python
        # merging datasets
        print('Join Table')
        pd.merge(df1,df2, on='id')

        #Result
        ```
        |   | id  | city  | birth_year | name   | math | english | sex | index_num |
        |---|-----|-------|-------------|--------|------|---------|-----|-----------|
        | 0 | 100 | Seoul | 1990        | Junho  | 90   | 80      | M   | 0         |
        | 1 | 101 | Pusan | 1989        | Heejin | 80   | 70      | F   | 1         |
        | 2 | 102 | Daegu | 1992        | Mijung | 70   | 60      | F   | 2         |

    -   #### FULL JOIN
        Use "outer" for the "how" parameter to perform a full (outer) join.
        ```python
        pd.merge(df1,df2, how='outer')
        #Result
        ``` 
        |    | id  | city      | birth_year | name    | math | english | sex | index_num |
        |----|-----|-----------|------------|---------|------|---------|-----|-----------|
        | 0  | 100 | Seoul     | 1990.0     | Junho   | 90.0 | 80.0    | M   | 0.0       |
        | 1  | 101 | Pusan     | 1989.0     | Heejin  | 80.0 | 70.0    | F   | 1.0       |
        | 2  | 102 | Daegu     | 1992.0     | Mijung  | 70.0 | 60.0    | F   | 2.0       |
        | 3  | 103 | Gangneung | 1997.0     | Minho   | NaN  | NaN     | NaN | NaN       |
        | 4  | 104 | Seoul     | 1982.0     | Steeve  | NaN  | NaN     | NaN | NaN       |
        | 5  | 105 | NaN       | NaN        | NaN     | 60.0 | 50.0    | M   | 3.0       |
        | 6  | 106 | Seoul     | 1991.0     | Mina    | NaN  | NaN     | NaN | NaN       |
        | 7  | 107 | NaN       | NaN        | NaN     | 50.0 | 40.0    | M   | 4.0       |
        | 8  | 108 | Pusan     | 1988.0     | Sumi    | NaN  | NaN     | NaN | NaN       |
        | 9  | 110 | Daegu     | 1990.0     | Minsu   | NaN  | NaN     | NaN | NaN       |
        | 10 | 111 | Gangneung | 1995.0     | Jinhee  | NaN  | NaN     | NaN | NaN       |
        | 11 | 113 | Seoul     | 1981.0     | Daeho   | NaN  | NaN     | NaN | NaN       |

    -   #### LEFT JOIN 
        Use "left" for the "how" parameter to perform a left join.
        ```python
        pd.merge(df1,df2,how='left')
        
        #Result
        ```
        | id  | city      | birth_year | name    | math | english | sex | index_num |
        |-----|-----------|------------|---------|------|---------|-----|-----------|
        | 100 | Seoul     | 1990       | Junho   | 90.0 | 80.0    | M   | 0.0       |
        | 101 | Pusan     | 1989       | Heejin  | 80.0 | 70.0    | F   | 1.0       |
        | 102 | Daegu     | 1992       | Mijung  | 70.0 | 60.0    | F   | 2.0       |
        | 103 | Gangneung | 1997       | Minho   | NaN  | NaN     | NaN | NaN       |
        | 104 | Seoul     | 1982       | Steeve  | NaN  | NaN     | NaN | NaN       |
        | 106 | Seoul     | 1991       | Mina    | NaN  | NaN     | NaN | NaN       |
        | 108 | Pusan     | 1988       | Sumi    | NaN  | NaN     | NaN | NaN       |
        | 110 | Daegu     | 1990       | Minsu   | NaN  | NaN     | NaN | NaN       |
        | 111 | Gangneung | 1995       | Jinhee  | NaN  | NaN     | NaN | NaN       |
        | 113 | Seoul     | 1981       | Daeho   | NaN  | NaN     | NaN | NaN       |

    -   #### Vertical concatenation
        The concat method allows you  to concatenate data vertically
        ```python
        data3 ={
            'id' : ['117','118','119','120','121'],
            'city' : ['Seoul','Pusan','Daegu','Gangneung','Seoul'],
            'birth_year' : [1990,1989,1992,1997,1982],
            'name' : ['Junho','Heejin','Mijung','Minho','Steeve']
        }

        df3 = DataFrame(data3)

        #concat
        concat_data = pd.concat([df1,df3])
        concat_data

        #Result
        ```
        | **id** | **city**    | **birth_year** | **name**  |
        |-----|-----------|------------|--------|
        | 100 | Seoul     | 1990       | Junho  |
        | 101 | Pusan     | 1989       | Heejin |
        | 102 | Daegu     | 1992       | Mijung |
        | 103 | Gangneung | 1997       | Minho  |
        | 104 | Seoul     | 1982       | Steeve |
        | 106 | Seoul     | 1991       | Mina   |
        | 108 | Pusan     | 1988       | Sumi   |
        | 110 | Daegu     | 1990       | Minsu  |
        | 111 | Gangneung | 1995       | Jinhee |
        | 113 | Seoul     | 1981       | Daeho  |
        | 117 | Seoul     | 1990       | Junho  |
        | 118 | Pusan     | 1989       | Heejin |
        | 119 | Daegu     | 1992       | Mijung |
        | 120 | Gangneung | 1997       | Minho  |
        | 121 | Seoul     | 1982       | Steeve |

    - #### Data manipulation and transformation

        - ###### data pivot
            pivoting is the process of transforming rows into columns and columns into rows
        ```python
        #hier_df 
        hier_df = DataFrame(
            np.arange(9).reshape((3,3)),
            index = [['a','a','b'],[1,2,2] ],
            columns = [['Pusan','Seoul','Pusan'] ,['Blue','Red','Red']]
        )

        #transform rows
        hier_df.stack()

        #Transform the rows labeled 'Blue' and 'Red' into columns.
        hier_df.stack().unstack()
        ```
        - ###### drop duplicates

        ```python

        #Create duplicates
        dupli_data = DataFrame({
            'col1' : [1,1,2,3,4,4,6,6],
            'col2' : ['a','b','b','b','c','c','b','b']
        })
        
        #The duplicated method is used to identify whether there are any duplicate entries in the data
        dupli_data.duplicated()

        #drop duplicates
        dupli_data.drop_duplicates()
        ```

        - ###### Mapping
            A feature that retriefes corresponding data from one dataset using key values shared with 
            another dataset
            ```python
            #ref data
            city_map ={
                'Seoul' : 'Sudo',
                'Gangenung' :'Yeondong',
                'Pusan' : 'Yeongnam',
                'Daegu' : 'Yeongnam'
            }
            
            df1['region'] = df1['city'].map(city_map)

            #Result
            ```
            | id  | city      | birth_year | name   | region    |
            |-----|-----------|------------|--------|-----------|
            | 100 | Seoul     | 1990       | Junho  | Sudo      |
            | 101 | Pusan     | 1989       | Heejin | Yeongnam  |
            | 102 | Daegu     | 1992       | Mijung | Yeongnam  |
            | 103 | Gangneung | 1997       | Minho  |           |
            | 104 | Seoul     | 1982       | Steeve | Sudo      |
            | 106 | Seoul     | 1991       | Mina   | Sudo      |
            | 108 | Pusan     | 1988       | Sumi   | Yeongnam  |
            | 110 | Daegu     | 1990       | Minsu  | Yeongnam  |
            | 111 | Gangneung | 1995       | Jinhee |           |
            | 113 | Seoul     | 1981       | Daeho  | Sudo      |

        - ###### Combining a lambda function with map
            ```python
            df1['up_two_num'] = df1['birth_year'].map(lambda x : str(x)[0:3])
            #Result
            ```
            | id  | city      | birth_year | name   | region    | up_two_num |
            |-----|-----------|------------|--------|-----------|------------|
            | 100 | Seoul     | 1990       | Junho  | Sudo      | 199        |
            | 101 | Pusan     | 1989       | Heejin | Yeongnam  | 198        |
            | 102 | Daegu     | 1992       | Mijung | Yeongnam  | 199        |
            | 103 | Gangneung | 1997       | Minho  |           | 199        |
            | 104 | Seoul     | 1982       | Steeve | Sudo      | 198        |
            | 106 | Seoul     | 1991       | Mina   | Sudo      | 199        |
            | 108 | Pusan     | 1988       | Sumi   | Yeongnam  | 198        |
            | 110 | Daegu     | 1990       | Minsu  | Yeongnam  | 199        |
            | 111 | Gangneung | 1995       | Jinhee |           | 199        |
            | 113 | Seoul     | 1981       | Daeho  | Sudo      | 198        |

        - ###### binning
            To aggregate the birth_year data in 5-year intervals, you need to perform binning
        ```python
        #binning
        birth_year_bins = [1980,1985,1990,1995,2000]

        birth_year_cut_data = pd.cut(df1.birth_year, birth_year_bins)
        #Result
        ```
        | id  | city      | birth_year | name   | region    | up_two_num | birth_year_bin |
        |-----|-----------|------------|--------|-----------|------------|----------------|
        | 100 | Seoul     | 1990       | Junho  | Sudo      | 199        | (1985, 1990]   |
        | 101 | Pusan     | 1989       | Heejin | Yeongnam  | 198        | (1985, 1990]   |
        | 102 | Daegu     | 1992       | Mijung | Yeongnam  | 199        | (1990, 1995]   |
        | 103 | Gangneung | 1997       | Minho  |           | 199        | (1995, 2000]   |
        | 104 | Seoul     | 1982       | Steeve | Sudo      | 198        | (1980, 1985]   |
        | 106 | Seoul     | 1991       | Mina   | Sudo      | 199        | (1990, 1995]   |
        | 108 | Pusan     | 1988       | Sumi   | Yeongnam  | 198        | (1985, 1990]   |
        | 110 | Daegu     | 1990       | Minsu  | Yeongnam  | 199        | (1985, 1990]   |
        | 111 | Gangneung | 1995       | Jinhee |           | 199        | (1990, 1995]   |
        | 113 | Seoul     | 1981       | Daeho  | Sudo      | 198        | (1980, 1985]   |
        ```python
        # naming
        group_names = ['early1980s','late1980s','early1990s','late1990s']
        birth_year_cut_data = pd.cut(df1.birth_year, birth_year_bins , labels=group_names)
        pd.value_counts(birth_year_cut_data)
        ```
        | birth_year   | count |
        |--------------|--------|
        | late1980s    | 4      |
        | early1990s   | 3      |
        | early1980s   | 2      |
        | late1990s    | 1      |


    - #### Data aggregation and group operations

        - ###### group by
        ```python
        df1.groupby('city').size()
        ```
        - ###### group operations
        ```python
        df1.groupby('city')['birth_yaer'].mean()
        ```
        - ###### iterator
            The variable group extracts the name of the region, and subdf extracts the rows coreesponding to that region
        ``` python
        for group , subdf in df1.groupby('region'):
        print("==============")
        print('Region Name:{0}'.format(group))
        print(subdf)
        ==============
        Region Name:Sudo
            id   city  birth_year    name region up_two_num
        0  100  Seoul        1990   Junho   Sudo        199
        4  104  Seoul        1982  Steeve   Sudo        198
        5  106  Seoul        1991    Mina   Sudo        199
        9  113  Seoul        1981   Daeho   Sudo        198
        ==============
        Region Name:Yeongnam
            id   city  birth_year    name    region up_two_num
        1  101  Pusan        1989  Heejin  Yeongnam        198
        2  102  Daegu        1992  Mijung  Yeongnam        199
        6  108  Pusan        1988    Sumi  Yeongnam        198
        7  110  Daegu        1990   Minsu  Yeongnam        199
        ```

    -   #### Handling missing data and outliers

        - ###### dropna
            Drop all rows with missing values
            ```python
            df.dropna()
            ```
            A method that ignores columns with missing values and uses only the available data
            ``` python
            df[[0,1]].dropna()
            ```
            Another method is to use "fillna()" to fill Nan with a specified value
            ```python
            df.fillna(0)
            ```
            The "ffill" method replaces missing values by propagating the last valid observation forward
            ```python
            df.fillna(method='ffill')
            ```

*   ### Basics of time series data analysis methods

    - #### Time sereis data manipulation and transformation
        ```python
        import pandas_datareader.data as pdr

        start_date = '2010-01-01'
        end_date = '2016-12-30'

        fx_jpusdata = pdr.DataReader('DEXJPUS', 'fred', start_date, end_date)
        fx_jpusdata.head()

        # Method to select exchange rates at the end of each month
        fx_jpusdata.resample('M').last().head()

        #Shift the data to calculate ratios
        #The index is unchanged, and the following code shifts the data down by one row
        fx_jpusdata.shift(1).head()

        # Comparison between today's and the previous day's exchange rate
        fx_jpusdata_ratio = fx_jpusdata / fx_jpusdata.shift(1)
        fx_jpusdata_ratio.head()
        ```
    
    - #### Moving average
        A 3-day moving average can be calculated using the fx_jpusdata data
        ```python
        fx_jpusdata.rolling(3).mean().head()
        # To calculate the trend of standard deviation instead of a moving average, use the std() method instead of mean()
        fx_jpusdata.rolling(3).std().head()
        ```
        










