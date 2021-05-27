from baseline_surrogate.demo_surrogate import check_surrogate_solution

if __name__ == '__main__':
    '''
    check_surrogate_solution checks the quality of a solution for an
    instance with 55 nodes. It requires a solution of the form
    [1, x1, x2, ..., x_n].
    The integers from 1 to n have to appear
    in the part [x1,..., x_n], so the number
    1 appears twice in total (1 is the starting depot).
    '''




    print('Check the quality of a bad solution...')
    solution = [1, 53, 14, 2, 41, 15, 36, 28, 44, 20, 11, 6, 46, 17, 55, 9, 32, 3, 8, 54, 27, 51, 52, 34, 49, 25, 1,
                31, 5, 37, 16, 43, 30, 29, 7, 12, 40, 22, 38, 21, 24, 35, 10, 39, 47, 26, 50, 13, 33, 19, 4, 23, 18,
                45, 42, 48]
    check_surrogate_solution(solution)

    print('Check the quality of a better solution...')
    solution = [1, 41, 9, 48, 52, 33, 50, 38, 12, 39, 11, 18, 13, 53, 5, 36, 6, 15, 17, 4, 2, 25, 1, 28, 26, 3, 23, 27,
                22, 55, 49, 54, 46, 16, 51, 21, 45, 44, 14, 24, 10, 34, 47, 31, 40, 19, 20, 8, 43, 30, 42, 32, 35, 37,
                29, 7]
    check_surrogate_solution(solution)

    print('Check the quality of an even better solution...')
    solution = [1, 11, 1, 41, 48, 52, 50, 33, 38, 12, 39, 18, 13, 53, 5, 36, 6, 15, 17, 4, 2, 25, 9, 28, 26, 3, 23, 27,
                22, 55, 49, 54, 46, 16, 51, 21, 45, 44, 14, 24, 10, 34, 47, 31, 40, 19, 20, 8, 43, 30, 42, 32, 35, 37,
                29, 7]
    check_surrogate_solution(solution)
