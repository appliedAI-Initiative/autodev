@main
def main(): Unit = {
  println(nth_fibonacci(7))
}

// An efficient, recursion-based function to calculate Fibonacci numbers.
def nth_fibonacci(n: Int): Int =
  @scala.annotation.tailrec
  <todo>
  proceed(n, 0, 1)
