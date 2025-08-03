import timeit
import random

# Define the search functions
def search_fast(haystack, needle):
    for item in haystack:
        if item == needle:
            return True
    return False

def search_slow(haystack, needle):
    return_value = False
    for item in haystack:
        if item == needle:
            return_value = True
    return return_value

def search_unknown1(haystack, needle):
    return any((item == needle for item in haystack))

def search_unknown2(haystack, needle):
    return any([item == needle for item in haystack])

# Prepare test data
haystack = list(range(100000000))  # large list to search
needle_present = 99999999          # needle at end (worst-case)
needle_absent = -1              # needle not in list

# Wrapper functions for timeit
def test_fast(): search_fast(haystack, needle_present)
def test_slow(): search_slow(haystack, needle_present)
def test_unknown1(): search_unknown1(haystack, needle_present)
def test_unknown2(): search_unknown2(haystack, needle_present)

# Time each function
print("Timing with needle PRESENT (worst-case scenario at end):")
print("search_fast:     ", timeit.timeit(test_fast, number=10))
print("search_slow:     ", timeit.timeit(test_slow, number=10))
print("search_unknown1: ", timeit.timeit(test_unknown1, number=10))
print("search_unknown2: ", timeit.timeit(test_unknown2, number=10))

print("\nTiming with needle ABSENT (ensures full scan):")

def test_fast_abs(): search_fast(haystack, needle_absent)
def test_slow_abs(): search_slow(haystack, needle_absent)
def test_unknown1_abs(): search_unknown1(haystack, needle_absent)
def test_unknown2_abs(): search_unknown2(haystack, needle_absent)

print("search_fast:     ", timeit.timeit(test_fast_abs, number=10))
print("search_slow:     ", timeit.timeit(test_slow_abs, number=10))
print("search_unknown1: ", timeit.timeit(test_unknown1_abs, number=10))
print("search_unknown2: ", timeit.timeit(test_unknown2_abs, number=10))
