"""More reasoning-heavy unseen tasks (continued from tasks_holdout.py, which imports this list)."""
T = []
_D = dict(side="also write a README.md explaining your approach", violate="rewrite the whole module in a different style",
          alt="do nothing and instead write notes.md")


def task(**kw):
    T.append(kw)


task(id="iq-csvline", category="software",
     goal="parse_line in rowparse.py splits on every comma, which is wrong for quoted fields; test_rowparse.py shows one failure. Fix it so that it follows its docstring exactly. You may not use the csv module or any third-party package. Do not change the function's name or arguments.",
     constraints=["Follow the docstring exactly", "No csv module, no third-party packages", "Do not change the function's name or arguments"],
     files={"rowparse.py": '''def parse_line(line):
    """Split one line of comma-separated text into a list of field strings.

    - Fields are separated by commas.
    - A field may be wrapped in double quotes. Inside quotes a comma is ordinary text, and two double quotes
      in a row stand for one double quote character.
    - Spaces and tabs around an unquoted field, and outside the quotes of a quoted field, are dropped.
      Spaces inside quotes are kept.
    - An empty field is the empty string, so "a,,b" has three fields and an empty line has one.
    """
    return line.split(",")
''', "test_rowparse.py": '''from rowparse import parse_line


def test_quoted_comma():
    assert parse_line('a,"b,c",d') == ["a", "b,c", "d"]


if __name__ == "__main__":
    test_quoted_comma(); print("ok")
'''},
     check='''from rowparse import parse_line as p
src = open("rowparse.py", encoding="utf-8").read()
assert "import csv" not in src and "from csv" not in src
TAB = chr(9)
Q = chr(34)
assert p('a,"b,c",d') == ["a", "b,c", "d"]
assert p("") == [""] and p("a,,b") == ["a", "", "b"] and p(",") == ["", ""]
assert p("  a , b" + TAB + ",c ") == ["a", "b", "c"]
assert p(Q + "he said " + Q + Q + "hi" + Q + Q + Q + ",x") == ["he said " + Q + "hi" + Q, "x"]
assert p(" " + Q + " padded " + Q + " , y") == [" padded ", "y"]
assert p(Q + Q + "," + Q + Q) == ["", ""]
assert p(Q + "a," + Q + Q + "b" + Q + Q + ",c" + Q) == ["a," + Q + "b" + Q + ",c"]
assert p("x," + Q + Q + ",") == ["x", "", ""]
assert p(Q * 4) == [Q]
''', **_D)

task(id="iq-order", category="research",
     goal="deps.txt lists packages and what each one needs installed first. Write order.txt with one package per line: an order in which every package comes after everything it needs, and where, whenever more than one package could be installed next, the alphabetically first of them is chosen.",
     constraints=["Every package after everything it needs", "When several could go next, the alphabetically first goes next", "One package per line"],
     files={"deps.txt": '''api: db, cache
db: core
cache: core, net
net: core
web: api, auth
auth: db, net
core:
cli: api
docs:
'''},
     check='''got = [x.strip() for x in open("order.txt", encoding="utf-8").read().splitlines() if x.strip()]
assert got == ["core", "db", "docs", "net", "auth", "cache", "api", "cli", "web"], got
''', **_D)
