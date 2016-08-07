import array

class matrix(object):

    def __init__(self, nrows, ncols, copy_from=None):
        self.nrows = nrows
        self.ncols = ncols

        self.num_elts = nrows*ncols

        self.arr = array.array(self.type)

        k = 0
        if copy_from is not None:
            while k < copy_from.num_elts:
                k += 1
                self.arr.append(copy_from[k])
        else:
            while k < self.num_elts:
                k += 1
                self.arr.append(0.0)

    def type_code(self):
        return self.arr.typecode

    def copy(self):
        this_copy = array.array(self.type)
        for elt in self.arr:
            this_copy.append(elt)
        return this_copy

    def offset(self, i, j):
        return self.col_offset(j) + i

    def col_offset(self, j):
        return j*self.nrows

    @staticmethod
    def LUsolve(P, LU, b):


    @staticmethod
    def forward_solve(LU, b):
        # Operate on L portion of LU matrix
        # L can be formed by copying all subdiagonal elements of LU, and placing 1 on the diagonal

        y = matrix(b.nrows, 1, b.type_code())

        y[0] = b[0]
        r = 1
        while r != b.nrows:
            y[r] = b[r]
            rr = 0
            while rr != r:
                y[r] -=



def main():

    nrows = 2
    ncols = 2

    mat = matrix(nrows, ncols, 'd')
    pass

if __name__ == '__main__':
    main()