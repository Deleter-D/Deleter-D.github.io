---
title: Java反射机制及注解在框架中的应用
tags:
  - Java
  - 反射
  - 注解
categories: Java
cover: https://cdn.jsdelivr.net/gh/Deleter-D/mycdn/blog-image/covers/java-reflection-and-annotation.png
abbrlink: 46441
date: 2022-05-15 23:09:02
---

# 反射

利用反射可以动态获取类的信息，并在程序运行期间动态创建对象，许多框架如spring、mybatis等均利用到了这一机制。

在编写代码或编译的过程中，可能无法得知要创建哪个对象，只有在运行时才能确定，这种情况下就需要利用反射机制，在运行时获取对象的各种信息。

利用一个例子帮助理解：

创建两个实体类；

```java
public class Pizaa {
    // 省略构造方法及Getter和Setter等方法
    private Integer id;
    private String type;
}
```

```java
public class Hamburger {
    // 省略构造方法及Getter和Setter等方法
    private Integer id;
    private String type;
}
```

创建一个配置文件，模拟spring等框架的配置文件，此处以properties配置文件为例；

```properties
# 指定要在运行时创建的类
bean=reflection.Pizaa
```

创建一个测试类；

```java
public class Test {

    private static Properties properties;

    static {
        try {
            properties = new Properties();
            // 获取类加载器，调用getResourceAsStream方法将配置文件作为流读入
            properties.load(Test.class.getClassLoader().getResourceAsStream("bean.properties"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception{
        // 获取配置文件中的参数
        String bean = properties.getProperty("bean");
        // 获取参数指定的类
        Class clazz = Class.forName(bean);
        // 获取该类的无参构造器
        Constructor constructor = clazz.getConstructor(null);
        // 利用构造器新建实例，此时已经获取到了配置文件中指定类型的实例
        Object target = constructor.newInstance(null);
        System.out.println(target);
    }
}
```

此时运行结果为：

```
Pizaa{id=null, type='null'}
```

如果将配置文件改为

```properties
bean=reflection.Hamburger
```

则运行结果为

```
Hamburger{id=null, type='null'}
```

利用这一机制可以实现一定程度上的解耦，无需用在编码时就指定要创建的实例类型，只需在配置文件中指定即可，使得类型的修改变得容易。

# 注解

注解需要结合反射来实现，注解本身只起到标记的作用，不进行实际的操作，实际操作由反射进行。

创建两个自定义注解，这里模拟Spring框架中的`@Component`注解和`@Value`注解；

**注意：此处自定义注解的名称和Spring框架中相同，是因为笔者并未引入Spring相关的依赖，故不会产生冲突。**

```java
// 如下两个注解用来描述该注解
// 指定该注解生效的时机，此处为运行时生效
@Retention(RetentionPolicy.RUNTIME)
// 指定标记的目标，此处表示该注解用来标记一个Java类型
@Target(ElementType.TYPE)
public @interface Component {
}
```

```java
@Retention(RetentionPolicy.RUNTIME)
// 此处表示该注解用来标记一个属性
// 可以通过这种格式指定多个目标：@Target({ElementType.TYPE,ElementType.FIELD})
@Target(ElementType.FIELD)
public @interface Value {
    // 利用一个方法来接收参数
    String value();
}
```

然后在实体类上打注解；

```java
@Component
public class Pizaa {
    @Value("1")
    private Integer id;
    @Value("bacon")
    private String type;
}
```

创捷另一个测试类；

```java
public class Test2 {

    private static Properties properties;

    static {
        try {
            properties = new Properties();
            // 获取类加载器，调用getResourceAsStream方法将配置文件作为流读入
            properties.load(Test.class.getClassLoader().getResourceAsStream("bean.properties"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        // 获取到目标类
        String bean = properties.getProperty("bean");
        Class clazz = Class.forName(bean);
        // 获取目标类的Component注解
        Annotation componentAnno = clazz.getAnnotation(Component.class);
        // 若该注解不为空则说明该类添加了该注解
        if (componentAnno != null) {
            // 该类添加了Component注解
            // 该注解的作用是创建对象，故获取该类的构造器
            Constructor constructor = clazz.getConstructor(null);
            // 利用构造器创建实例
            Object target = constructor.newInstance(null);

            // 处理Value注解
            // 获取该类所有的属性，但不包括继承得来的属性，仅有该类自身的属性
            Field[] declaredFields = clazz.getDeclaredFields();
            for (Field declaredField : declaredFields) {
                Value valueAnnoOnId = declaredField.getAnnotation(Value.class);
                if (valueAnnoOnId != null) {
                    // 该属性添加了Value注解
                    // 通过调用注解中定义的方法即可取得参数
                    String value = valueAnnoOnId.value();
                    // 暴力反射机制，设置为ture，则可以强行给private修饰的属性赋值
                    declaredField.setAccessible(true);
                    // 处理属性的类型问题
                    switch (declaredField.getType().getName()) {
                        // 可以添加多个case以处理不同类型
                        case "java.lang.Integer":
                            Integer val = Integer.parseInt(value);
                            // 通过set方法将value的值赋给target对象的该属性
                            declaredField.set(target, val);
                            break;
                        default:
                            declaredField.set(target, value);
                            break;
                    }
                }
            }
            System.out.println(target);
        } else {
            // 该类未添加Component注解
            System.out.println("无法创建" + clazz.getName() + "对象");
        }
    }
}
```

此时运行结果为：

```
Pizaa{id=1, type='bacon'}
```

若将实体类上的注解注释掉；

```java
// @Component
public class Pizaa {
    @Value("1")
    private Integer id;
    @Value("bacon")
    private String type;
}
```

则运行结果为：

```
无法创建reflection.Pizaa对象
```

通过上述例子可以看到，注解并不进行任何实际的操作，仅仅作为标记作用，而实际的操作需要通过反射机制，在运行时获取目标类后进行判断，若不为空则说明添加了该注解，而后进行一系列的业务处理。
