# tokenizecode

This package provides an easy interface to tree_sitter machine learning on source code.

It parses source code into a syntax tree, in which every leaf actually occurs in the code file:

## Usage

This package provides a parser and a BPE implementation, that applies BPE to leaves
in the tree.

### Parsing

```python
from tokenizecode import CodeParser

java_code = """class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!"); 
    }
}"""

parser = CodeParser()

# returns a tensortree
tree = parser.parse(java_code, "java")

tree.pprint()
```
