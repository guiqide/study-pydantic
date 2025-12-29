import functools


class Student(object):
  pass


def log(func):
  @functools.wraps(func)
  def wrapper(*args, **kw):
    print('call %s():' % func.__name__)
    return func(*args, **kw)
  return wrapper

def logDecorator(text):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
      print('%s %s():' % (text, func.__name__))
      return func(*args, **kw)
    return wrapper
  return decorator

@logDecorator('execute')
def now():
  print('2015-3-25')

f = now
f()
print(f.__name__)
