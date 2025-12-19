import os
import sys

sys.path.append(os.path.abspath("../core/build/"))

import minigrad


def main():
    a = minigrad.Value(2)
    b = minigrad.Value(3)

    c = a + b
    d = a - b

    e = c * d

    f = e / a

    print("a =", a)
    print("b =", b)
    print("c = a + b =", c)
    print("d = a - b =", d)
    print("e = c * d =", e)
    print("f = e / a =", f)


if __name__ == "__main__":
    main()
