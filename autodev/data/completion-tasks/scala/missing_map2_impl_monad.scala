trait Monad[F[_]]:
  def bind[A](a: => A): F[A]

  extension[A] (ffa: F[F[A]]) def join: F[A] =
    ffa.flatmap(identity)

  extension [A](fa: F[A])
    def flatmap[B](fab: A => F[B]): F[B] =
      fa.map(fab).join

    def map[B](f: A => B): F[B] =
      fa.flatmap(a => bind(f(a)))

    def map2[B, C](fb: F[B])(f: (A, B) => C): F[C] =
      <todo>

  def sequence[A](fas: List[F[A]]): F[List[A]] =
    fas.foldRight(bind(List[A]()))((fa, acc) => fa.map2(acc)(_ :: _))

  def traverse[A, B](as: List[A])(f: A => F[B]): F[List[B]] =
    as.foldRight(bind(List[B]()))((a, acc) => f(a).map2(acc)(_ :: _))
