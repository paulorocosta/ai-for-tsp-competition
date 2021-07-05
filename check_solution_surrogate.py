from baseline_surrogate.demo_surrogate import check_surrogate_solution

if __name__ == '__main__':
    '''
    check_surrogate_solution checks the quality of a solution for an
    instance with 65 nodes. It requires a solution of the form
    [1, x1, x2, ..., x_n].
    The integers from 1 to n have to appear
    in the part [x1,..., x_n], so the number
    1 appears twice in total (1 is the starting depot).
    '''



    print('Check the quality of a bad solution...')
    solution = [1, 65, 14, 57, 6, 50, 25, 29, 5, 7, 53, 35, 2, 51, 36, 28, 26, 1, 64, 44, 4, 49, 45, 48, 3, 21, 55, 9, 61, 33, 15, 34, 58, 63, 47, 60, 40, 59, 8, 43, 12, 46, 31, 10, 37, 24, 19, 17, 20, 52, 62, 42, 54, 32, 18, 38, 30, 56, 13, 16, 23, 11, 41, 39, 27, 22]
    check_surrogate_solution(solution)

    print('Check the quality of a better solution...')
    solution = [1, 1, 2, 3, 4, 5, 65, 13, 36, 46, 48, 16, 56, 31, 6, 26, 64, 63, 21, 62, 7, 61, 8, 60, 11, 9, 10, 59, 12, 30, 32, 14, 58, 15, 17, 57, 55, 18, 19, 20, 54, 53, 22, 23, 52, 24, 51, 25, 27, 28, 50, 49, 29, 33, 47, 45, 44, 34, 35, 43, 42, 37, 38, 41, 40, 39]
    check_surrogate_solution(solution)

    print('Check the quality of an even better solution...')
    solution = [1, 6, 1, 32, 28, 46, 59, 55, 42, 39, 18, 64, 40, 41, 51, 58, 35, 7, 23, 45, 2, 63, 4, 57, 37, 11, 27, 31, 9, 56, 61, 50, 3, 8, 12, 60, 53, 33, 13, 36, 47, 15, 29, 19, 5, 22, 48, 26, 44, 25, 62, 16, 17, 21, 65, 38, 49, 14, 34, 20, 54, 43, 52, 24, 10, 30]
    check_surrogate_solution(solution)

