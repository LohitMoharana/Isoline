"""Minimal KiCad netlist (.net, s-expression) reader: components {ref: (value, footprint, description)} and
nets {name: [(ref, pin), ...]}."""
import re


def sexpr(text):
    tok = re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()]+', text)
    stack = [[]]
    for t in tok:
        if t == "(":
            stack.append([])
        elif t == ")":
            x = stack.pop(); stack[-1].append(x)
        else:
            stack[-1].append(t[1:-1] if t.startswith('"') else t)
    return stack[0][0]


def find(node, key):
    return [c for c in node if isinstance(c, list) and c and c[0] == key]


def val(node, key, default=""):
    f = find(node, key)
    return f[0][1] if f and len(f[0]) > 1 else default


def read(path):
    root = sexpr(open(path).read())
    comps = {}
    for c in find(find(root, "components")[0], "comp"):
        comps[val(c, "ref")] = (val(c, "value"), val(c, "footprint"), val(c, "description"))
    nets = {}
    for n in find(find(root, "nets")[0], "net"):
        nets[val(n, "name")] = [(val(x, "ref"), val(x, "pin")) for x in find(n, "node")]
    return comps, nets
