from fdth import fdt

# Tests for the fdt function

x1 = ["A", "B", "A", "C", "B", "B", "C", "A", "C", "C"]
x2 = ["X", "Y", "X", "Z", "Y", "Y", "Z", "Z", "Z", "Z"]
x3 = ["Red", "Blue", "Green", "Red", "Green", "Blue", "Blue", "Green"]

# Test 1
print("Test 1 - Python")
print(fdt(x1, sort=True, decreasing=True))

# Test 2
print("\nTest 2 - Python")
print(fdt(x2, sort=False, decreasing=False))

# Test 3
print("\nTest 3 - Python")
print(fdt(x3, sort=True, decreasing=False))
