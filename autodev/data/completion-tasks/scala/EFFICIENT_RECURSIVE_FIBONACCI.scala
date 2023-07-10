// Simple code that's based on https://github.com/fpinscala/fpinscala, but can be found in many functional codebases.

@main
def main(): Unit = {
  println(nth_fibonacci(7))
}

// An efficient, recursion-based function to calculate Fibonacci numbers.
def nth_fibonacci(n: Int): Int =
  @scala.annotation.tailrec
  def proceed(n: Int, a: Int, b: Int): Int =
    if n <= 0 then a
    else proceed(n - 1, b, a + b)
  proceed(n, 0, 1)

<todo>
