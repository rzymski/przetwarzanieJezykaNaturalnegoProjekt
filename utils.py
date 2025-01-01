import time


def measureExecutionTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        executionTime = endTime - startTime
        print(f"⏱️ Funkcja '{func.__name__}' wykonana w {executionTime:.4f} sekund.")
        return result
    return wrapper
