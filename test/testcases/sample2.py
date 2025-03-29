class SampleClass:
    def method_with_docstring(self):
        """Existing docstring."""
        return "Hello Class"

    def method_without_docstring(self):
        return "Hello again"

def myadd(a, b):
    return a + 2 * b

def mysub(a, b):
    """Subtract two numbers."""
    return a - b

class MyClass:
    def mymethod(self,
                 a, b):
        return a * b

    def mymethod2(self,
                  a, b):
        """
        Multiply two numbers. Whoa!

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int: The product of a and b.
        """
        return a / b
