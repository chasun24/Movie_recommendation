
class A:
    def __init__(self):
        print("created")
        pass
    def a(self):
        print("3333")
        return 3

if __name__=="__main__":
    tt=A()
    n=tt.a()
    print(n)