import pandas as pd

def test_columns_and_no_nulls():
    df = pd.read_csv('data/iris.csv')
    expected = {'sepal_length','sepal_width','petal_length','petal_width','species'}
    assert set(df.columns) == expected
    assert not df.isnull().any().any()
