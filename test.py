

class KFunction:
    def __init__(self, name: str|None=None):
        self.name = name

    def call(self, *args, **kwargs):
        print("Calling", self.name)
        return self.name
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

class KStruct:
    def __init__(self):
        for name, value in self.__class__.__dict__.items():
            if isinstance(value, KFunction):
                value.name = name

#Basic version
class MyClass(KStruct):

    my_function = KFunction()

    def __init__(self):
        super().__init__()

#Mechanism to wrap purely by name
class MyClass(KStruct):

    __my_function = KFunction("my_function")

    def __init__(self):
        super().__init__()

    def my_function(self):
        print("Before")
        res = self.__my_function()
        print("After")
        return res

c = MyClass()
res = c.my_function()
print(res)

