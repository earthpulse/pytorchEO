class ClassA():
    """ This is Class A """

    def __init__(self):
        self.name = "class A"

    def sayHi(self):
        """ say hi """
        print(f'{self.name} says hi')


class ClassB(ClassA):
    """ This is Class B """

    def __init__(self):
        self.name = "class B"
