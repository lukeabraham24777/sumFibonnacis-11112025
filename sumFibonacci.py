
def fibonacci(fn1=1,fn2=0,arr=[0,1]): #fn1 is f_{n-1}, fn2 is f_{n-2}
    if fn1 + fn2 > 11112025:
        return arr
    arr.append(fn1+fn2)
    return fibonacci(fn1 + fn2, fn1, arr)
    
x = fibonacci()

print(sum(x))