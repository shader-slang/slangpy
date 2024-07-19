


class MyClass:
    my_function = lambda x: x + 1

    def __init__(self, value):
        set.__setattr__(self, 'my_function', lambda x: x + value)
        pass
        #self.my_function = lambda x: x + value

print(MyClass.my_function(1))  # 2

c = MyClass(10)
print(c.my_function(1))

