# 类的继承
class Person:#基类
    def __init__(self,name,age,gender):
        self.name=name
        self.age=age
        self.gender=gender
        # __init__是构造方法，在创建对象时自动调用，用于初始化对象的属性

    def personInfo(self):#self为类的实例本身，通过self访问类的属性和方法
        print(f"name:{self.name}")
        print(f"age:{self.age}")
        print(f"gender{self.gender}")

class Student(Person):#派生类
    def __init__(self,name,age,gender,college,class_name):
        super().__init__(name,age,gender)#这是调用父类构造方法的
        self.college=college
        self.class_name=class_name

    def personInfo(self):
        super().personInfo()#调用父类方法
        print(f"college:{self.college}")
        print(f"class:{self.class_name}")

    def __str__(self):
        return f"name={self.name} age={self.age} gender={self.gender} college={self.college} class={self.class_name}"
    
if __name__=="main":
    person=Person("ZhanRuixuan",18,"boy")
    person.personInfo()
    student=Student("someone",19,"girl","cv","class1")
    student.personInfo()
    print("use str method")
    print(student)