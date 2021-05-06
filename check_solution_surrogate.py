from baseline_surrogate.demo_surrogate import check_surrogate_solution

if __name__ == '__main__':
    '''
    check_surrogate_solution checks the quality of a solution for an
    instance with 50 nodes. It requires a solution of the form
    [1, x1, x2, ..., x_n].
    The integers from 1 to n have to appear
    in the part [x1,..., x_n], so the number
    1 appears twice in total (1 is the starting depot).
    '''

    print('Check the quality of a bad solution...')
    solution = [1, 24, 15, 36, 10, 20, 17, 40, 30, 5, 4, 21, 6, 19, 11, 43, 33, 28, 41, 12, 2, 44, 50, 3, 39, 25, 16, 1,
                42, 8, 31, 35, 46, 7, 49, 47, 23, 37, 45, 32, 13, 29, 22, 34, 38, 26, 14, 9, 27, 48, 18]

    check_surrogate_solution(solution)

    print('Check the quality of a good solution...')
    solution = [1, 45, 7, 29, 16, 36, 37, 2, 21, 26, 39, 32, 27, 42, 5, 30, 22, 47, 46, 34, 28, 43, 15, 33, 6, 49, 18,
                38, 23, 41, 8, 20, 44, 14, 24, 31, 25, 11, 4, 35, 17, 12, 3, 10, 40, 50, 13, 9, 1, 48, 19]

    check_surrogate_solution(solution)
