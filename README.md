tinynnumpy
==========

Minimal, pure Python reimplementation of the NumPy N-dimensional array class.

This was an attempt to emulate the NumPy ndarray slicing behavior, built on top of a flat array of floating point data, which I found fascinating.

I'm also planning to use it in my Advanced CSV Sublime Text plugin as a fallback in case NumPy isn't installed.  (It's obviously way over-engineered for that purpose)

# Basic usage

```python
> import tinyndarray
> m = array([[1,2],[3,4]])
> m
array(
[
	[1.0, 2.0], 
	[3.0, 4.0]])
```

# Basic slicing

```python
> m[1:2,0:2]
array(
[
	[3.0, 4.0]])
```

# Slicing with step

```python
> m = array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
> m[1::2,::3]
array(
[
	[5.0, 8.0], 
	[13.0, 16.0]])
```
