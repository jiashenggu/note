# datasets

## indexed_dataset

This module converts between Python values and C structs represented as Python bytes objects.  
https://docs.python.org/3/library/struct.html

A memoryview object exposes the C level buffer interface as a Python object which can then be passed around like any other object.  
https://docs.python.org/3/c-api/memoryview.html

Create a memory-map to an array stored in a binary file on disk.

Memory-mapped files are used for accessing small segments of large files on disk, without reading the entire file into memory. NumPy’s memmap’s are array-like objects. This differs from Python’s mmap module, which uses file-like objects.
https://numpy.org/doc/2.1/reference/generated/numpy.memmap.html
