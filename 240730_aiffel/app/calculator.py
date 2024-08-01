def calculate(x, y, operator):
    """
    두 수에 대한 기본적인 사칙연산 수행
    : 곱하기, 나누기, 더하기, 빼기
    return: 두 수의 사칙연산 결과
    """
    if operator == "+":
        val = x + y
    elif operator == "-":
        val = x - y
    elif operator == "/":
        if y == 0:
            val = "y must not be zero"
        else:
            val = x / y
    elif operator == "*":
        val = x * y
    return val