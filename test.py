import sys
def test(x,*args):
    print(args)

if __name__ == "__main__":
    args = sys.argv
    test(1,*args)