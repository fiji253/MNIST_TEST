import torch as ptc

class MatMulTest (ptc):

   def __init__(self):
      self.A = ptc.Tensor([[1, 2], [3, 4]])
      self.B = ptc.Tensor([[5, 6], [7, 8]])


   def matmulmeth(self):
      meth1 = ptc.matmul(self.A, self.B)
      meth2 = self.A @ self.B
      meth3 = ptc.mm(self.A, self.B)


      return meth1

#test1 = MatMulTest.matmulmeth()
test = MatMulTest
result = test.matmulmeth()
print(result) 