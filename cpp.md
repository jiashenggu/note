在 C++ 中，static 和 const 是两个关键字，它们的作用和用途有很大的不同。以下是它们的详细说明：
------------------------
## static 的作用
static 关键字在 C++ 中有多种用途，主要与变量、函数和类成员的生命周期、作用域以及存储方式有关。

(1) 静态变量（Static Variable）
•  定义：在函数内部声明的静态变量。
•  作用：静态变量的生命周期是全局的，但作用域是局部的。它不会在函数调用结束后被销毁，而是保留其值，直到程序结束。
•  示例： 
void func() {
    static int count = 0; // 每次调用 func()，count 的值都会保留
    count++;
    std::cout << "count = " << count << std::endl;
}

 每次调用 func() 时，count 的值都会递增，而不是重新初始化为 0。
 
(2) 静态成员变量（Static Member Variable）
•  定义：在类中声明的静态成员变量。
•  作用：静态成员变量属于类本身，而不是类的实例。所有类的实例共享同一个静态成员变量。
•  示例： 
```c++
class MyClass {
public:
    static int count; // 声明静态成员变量
};
int MyClass::count = 0; // 定义静态成员变量
```
 所有 MyClass 的实例共享 count，可以通过 MyClass::count 直接访问。
 
(3) 静态成员函数（Static Member Function）
•  定义：在类中声明的静态成员函数。
•  作用：静态成员函数不能访问非静态成员变量或非静态成员函数，因为它没有 this 指针。它通常用于处理与类相关的静态数据。
•  示例： 
```cpp
class MyClass {
public:
    static void func() {
        // 只能访问静态成员变量
        std::cout << "This is a static member function." << std::endl;
    }
};
MyClass::func(); // 调用静态成员函数
```

(4) 静态库函数（Static Library Function）
•  定义：在 C++ 中，static 也可以用于表示一个函数是静态的，通常用于限制其作用域。
•  作用：静态函数只能在定义它的文件中使用，不能被其他文件调用。
•  示例： 
```cpp
static void helper() { // 该函数只能在当前文件中使用
    std::cout << "This is a static function." << std::endl;
}
```
------------------------
## const 的作用
const 关键字用于声明常量，表示某个值在程序运行期间不能被修改。
(1) 常量变量（Constant Variable）
•  定义：用 const 声明的变量。
•  作用：常量变量的值在初始化后不能被修改。
•  示例： 
```cpp
const int MAX_VALUE = 100; // MAX_VALUE 的值不能被修改
```
(2) 常量指针（Constant Pointer）
•  定义：用 const 声明的指针。
•  作用：指针指向的内存地址的内容不能被修改。
•  示例： 
```cpp
const int* ptr = &value; // ptr 指向的值不能被修改
```
(3) 指向常量的指针（Pointer to Constant）
•  定义：用 const 声明的指针。
•  作用：指针本身可以改变，但指向的值不能被修改。
•  示例：
```cpp
int value = 10;
const int* ptr = &value; // ptr 指向的值不能被修改，但 ptr 可以指向其他地址
```
(4) 常量引用（Constant Reference）
•  定义：用 const 声明的引用。
•  作用：引用指向的值不能被修改。
•  示例： 
```cpp
const int& ref = value; // ref 指向的值不能被修改
```
(5) 常量成员函数（Constant Member Function）
•  定义：在类中用 const 声明的成员函数。
•  作用：常量成员函数不能修改类的非静态成员变量。
•  示例： 
```cpp
class MyClass {
public:
    int getValue() const { // 常量成员函数
        return value; // 只能读取 value，不能修改
    }
private:
    int value;
};
```
static 和 const 的区别

| 特性 | static | const |
|---|---|---|
| 作用域 | 静态变量的作用域是局部的，但生命周期是全局的。静态成员变量属于类，所有实例共享。 | const 用于声明常量，限制变量的值不能被修改。作用域取决于变量的定义位置。 |
| 生命周期 | 静态变量的生命周期是全局的，直到程序结束。 | const 变量的生命周期与普通变量相同，但值不能被修改。 |
| 可变性 | 静态变量可以被修改（除非被 const 修饰）。 | const 变量的值不能被修改。 |
| 作用对象 | 可以修饰变量、函数、成员变量、成员函数。 | 可以修饰变量、指针、引用、成员函数。 |
| 默认值 | 静态变量默认初始化为 0（对于内置类型）。 | const 变量必须在定义时初始化（对于内置类型）。 |
| 组合使用 | static 和 const 可以组合使用。例如：static const int VALUE = 10; 表示一个全局常量。 | const 可以与 static 组合使用，也可以单独使用。 |

总结
•  static 主要用于控制变量或函数的生命周期和作用域，通常用于共享数据或限制作用域。
•  const 主要用于声明常量，限制变量的值不能被修改，以提高代码的安全性和可读性。
两者在功能和用途上有很大的不同，但在某些情况下可以组合使用，以实现特定的需求。
