import pytest
import polars
from pymovements.transforms import value_impossible

# check if wrong mode raises a ValueError
def test_value_impossible_invalid():
	with pytest.raises(ValueError):
		value_impossible(maxi=230, mode="bad", input_column="hello", output_column="bye",num=2)

# check for the clip mode
def test_value_impossible_clip():
	res = value_impossible(maxi=230, mode="clip", input_column="hello", output_column="bye",num=2)

	expr = polars.concat_list([
		polars.col("hello").list.get(0).clip(-230, 230),
		polars.col("hello").list.get(1).clip(-230, 230),
		]).alias("bye")

	assert str(expr) == str(res)


# check for the null mode
def test_value_impossible_null():
	res = value_impossible(maxi=230, mode="null", input_column="hello", output_column="bye",num=2)

	expr = polars.concat_list([
		polars.when(polars.col("hello").list.get(0).abs() <= 230).then(polars.col("hello").list.get(0)).otherwise(None),
		polars.when(polars.col("hello").list.get(1).abs() <= 230).then(polars.col("hello").list.get(1)).otherwise(None),
		]).alias("bye")

	assert str(expr) == str(res)