#!/usr/local/bin python3

import mariadb
import numpy
import sys
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split


def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']


def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse

def update(params, grads):
    return jax.tree_map(lambda p, g: p - 0.05 * g, params, grads)

def main():
  try:
     conn = mariadb.connect(
        user="root",
        password="nestor123",
        host="localhost",
        port=3306,
        database="Dinero"
     )
  except mariadb.Error as e:
    print(f'error with {e}')
    sys.exit(1)

  cur = conn.cursor()
  cur.execute(
     '''
     SELECT age,`hours-per-week` , case salary when ' <=50K' then 0 
     when ' >50K' then 1 else -1 end 
     FROM salary;
     '''
  )

  X = []
  y = []

  for (age, hours, salary) in cur:
    X.append([age, hours])
    y.append(salary)

  X = numpy.array(X)
  y = numpy.array(y)

  X, X_test, y , y_test = train_test_split(X,y)

  # model weights
  params = {
     'w': jnp.zeros(X.shape[1:]),
     'b': 0.
  }

  grad_fn = jax.grad(loss_fn)

  # the main training loop
  for _ in range(50):
    loss = loss_fn(params, X_test, y_test)
    print(loss)

    grads = grad_fn(params, X, y)
    params = update(params, grads)

if __name__ == '__main__':
    main()