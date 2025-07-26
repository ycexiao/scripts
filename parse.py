import re
from pathlib import Path
import libcst as cst

def get_docstring_from_node(node):
    """
    Returns the docstring string if node.body starts with a string expression,
    otherwise returns None.
    """
    # node.body is usually an IndentedBlock for classes/functions, or list for Module
    body = node.body
    if isinstance(body, cst.IndentedBlock):
        stmts = body.body  # This is a list of statements
    elif isinstance(body, list):
        stmts = body
    else:
        # Unknown type, can't extract docstring
        return None

    if stmts:
        first_stmt = stmts[0]
        if (
            isinstance(first_stmt, cst.SimpleStatementLine)
            and len(first_stmt.body) == 1
            and isinstance(first_stmt.body[0], cst.Expr)
            and isinstance(first_stmt.body[0].value, cst.SimpleString)
        ):
            raw_value = first_stmt.body[0].value.value
            if (raw_value.startswith('"""') and raw_value.endswith('"""')) or (
                raw_value.startswith("'''") and raw_value.endswith("'''")
            ):
                return raw_value[3:-3]
            else:
                return raw_value[1:-1]
    return None

class DocstringCollector(cst.CSTVisitor):
    def __init__(self):
        self.scope = []
        self.docstrings = {}

    def visit_Module(self, node):
        doc = get_docstring_from_node(node)
        if doc:
            self.docstrings['module'] = doc

    def visit_ClassDef(self, node):
        self.scope.append(node.name.value)
        doc = get_docstring_from_node(node)
        if doc:
            qualname = ".".join(self.scope)
            self.docstrings[qualname] = doc

    def leave_ClassDef(self, original_node):
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name.value)
        doc = get_docstring_from_node(node)
        if doc:
            qualname = ".".join(self.scope)
            self.docstrings[qualname] = doc

    def leave_FunctionDef(self, original_node):
        self.scope.pop()


def extract_docstrings(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = cst.parse_module(source)
    collector = DocstringCollector()
    tree.visit(collector)
    return collector.docstrings


class DocstringRewriter(cst.CSTTransformer):
    def __init__(self, docstring_map):
        self.docstring_map = docstring_map
        self.scope = []

    def visit_ClassDef(self, node):
        self.scope.append(node.name.value)

    def leave_ClassDef(self, original_node, updated_node):
        self.scope.pop()
        return updated_node

    def visit_FunctionDef(self, node):
        self.scope.append(node.name.value)

    def leave_FunctionDef(self, original_node, updated_node):
        self.scope.pop()
        return updated_node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        if (
            original_node.body
            and isinstance(original_node.body[0], cst.Expr)
            and isinstance(original_node.body[0].value, cst.SimpleString)
        ):
            qualified_name = ".".join(self.scope) if self.scope else "module"
            if qualified_name in self.docstring_map:
                new_doc = self.docstring_map[qualified_name]
                new_string = cst.SimpleString(f'"""{new_doc}"""')
                new_expr = original_node.body[0].with_changes(value=new_string)
                return updated_node.with_changes(body=[new_expr])
        return updated_node

def rewrite_docstrings(filepath, docstring_map):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = cst.parse_module(source)
    rewriter = DocstringRewriter(docstring_map)
    modified_tree = tree.visit(rewriter)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(modified_tree.code)

def build_numpy_section(title, items):
    indent_levels = list(map(lambda l: len(l) - len(l.lstrip()), [t[0] for t in items]))
    lowest_indent = min([lvl for lvl in indent_levels if lvl > 0], default=0)
    output = []
    if title:
        output.append(' ' * lowest_indent + title)
        output.append(' ' * lowest_indent + len(title) * '-')
    else:
        output.append('')
        output.append(' '*lowest_indent + 'Attributes')
        output.append(' ' * lowest_indent + '----------')
    for name, desc in items:
        desc_lines = desc.split('\n')
        desc_formatted = ('\n' + ' ' * (lowest_indent + 4)).join(desc_lines)
        desc_formatted = " "*(lowest_indent + 4) + desc_formatted.strip()
        output.append(f'{name}')
        output.append(f'{desc_formatted}')
    return output


def parse_custom_sections(docstring):
    lines = docstring.strip().splitlines()
    output = []
    section_lines = []
    current_section = None
    attr_pattern = re.compile(r'^(\s*\S+)\s*--\s*(.*)$')

    i = 0
    in_section = False
    while i < len(lines):
        line = lines[i].rstrip()

        if in_section:
            m = attr_pattern.match(line)
            if m:
                section_lines.append([m.group(1).rstrip(':'), m.group(2).rstrip()])
            else:
                if section_lines:
                    section_lines[-1][1] += '\n' + line.strip()

            # End of a section
            if (i == len(lines) - 1 or lines[i + 1].strip() == '') and in_section:
                # Flush previous section
                output.extend(build_numpy_section(current_section, section_lines))
                section_lines = []
                in_section = False

        # Start of a new section
        elif i != len(lines) - 1 and lines[i+1].count('--') == 1:
            if line != '':
                current_section = line.strip().rstrip(':')
            else:
                current_section = None
            in_section = True

        else:
            output.append(line)

        i += 1

    total_output = '\n'.join(output)
    if len(output) > 1:
        indent_levels = list(map(lambda l: len(l) - len(l.lstrip()), output))
        lowest_indent = min([lvl for lvl in indent_levels if lvl > 0], default=0)
        total_output += '\n'+' ' * lowest_indent  # maintain indentation

    return total_output

    
if __name__ == "__main__":
    # want_break = False
    source_dir_path = Path("diffpy.srfit/src/diffpy/srfit/")
    for file in source_dir_path.rglob("*.py"):
        docs = extract_docstrings(str(file))
        for qualname, doc in docs.items():
            docs[qualname] = parse_custom_sections(doc)
        rewrite_docstrings(str(file), docs)