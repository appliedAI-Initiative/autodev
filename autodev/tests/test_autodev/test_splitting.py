from autodev.splitting import get_ast_chunks, split_into_semantic_chunks

module_code = '''
import somelib
import some_other_lib


class A:
    """
    This is class A
    """
    def method1(self):
        """This is method1"""
        # and this is a comment
        pass

    def method2(self):
        """This is method2"""
        pass

    
class B:
    # block comment in class
    """This is class B"""
    def method3(self, a: int, b: Optional[str] = None):
        """This is method3"""
        pass

    def method4(self):
        """This is method4"""
        pass
        
some_global = 1

# block comment in main program

"""
Block comment as multiline
string
"""

"Block comment as single line string"

"""Block comment as multiline one-line string"""

for a in range(10):
    # block comment in for loop
    for_loop(a)

    
def func1():
    """This is func1"""
    import import_in_func
    # block comment in function
    pass

def func2():
    """This is func2"""
    pass


if __name__ == '__main__':
    """This is the main program"""
    a = A()
    a.method1()
    a.method2()
    b = B()
    b.method3()
    b.method4()
    func1()
    func2()
'''


class TestSplitting:
    def test_semantic_chunking_basics(self):
        all_chunks = get_ast_chunks(module_code)
        aggregated_chunks = split_into_semantic_chunks(
            module_code, num_lines=30, max_lines=100
        )
        assert len(all_chunks) > len(aggregated_chunks)
        # docstrings were extracted
        assert "This is the main program" in aggregated_chunks[-1]
