#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
 */

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high)
{
    double range = (high - low + 0);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

void create_rand_matrix(matrix *result, double l, double h)
{
    for (int i = 0; i < result->rows; i++)
    {
        for (int k = 0; k < result->cols; k++)
        {
            set(result, i, k, rand_double(l, h));
        }
    }
}


/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high)
{
    srand(seed);
    create_rand_matrix(result, low, high);
}

// HELPER
void complete_alloc_matrix(matrix **mat, int rows, int cols, double **p_data, double *data, double *tp)
{
#pragma omp parallel for if (rows > 500)
    for (int i = 0; i < rows; i++)
    {
        p_data[i] = data + (i * cols);
    }

    (*mat)->data = p_data;
    (*mat)->tp = tp;
    (*mat)->cols = cols;
    (*mat)->rows = rows;
    (*mat)->is_1d = (rows == 1 || cols == 1);
    (*mat)->parent = NULL;

    // change?
    // some people say it should point to a value if shared
    (*mat)->ref_cnt = 1;
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fields of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols)
{
    if (rows < 0 || rows <= 0)
    {
        fprintf(stderr, "Bad value for rows: %d\n", rows);
        return -1;
    }
    else if (cols <= 0)
    {
        fprintf(stderr, "Bad value for cols: %d\n", cols);
        return -1;
    }

    *mat = malloc(sizeof(matrix));
    if (!mat)
    {
        fprintf(stderr, "Malloc for matrix failed.\n");
        return -1;
    }

    double **p_data = malloc(rows * sizeof(double *));
    if (!p_data)
    {
        fprintf(stderr, "Malloc for matrix rows failed.\n");
        return -1;
    }
    int mat_size = rows * cols;
    double *data = calloc(mat_size, sizeof(double));
    if (!data)
    {
        fprintf(stderr, "Calloc for matrix data failed.\n");
        return -1;
    }
    double *tp = malloc(mat_size * sizeof(double));
    if (!tp)
    {
        fprintf(stderr, "Malloc for matrix transpose failed.\n");
        return -1;
    }

    complete_alloc_matrix(mat, rows, cols, p_data, data, tp);

    return 0;
}

// HELPER
void allocate_matrix_ref_helper(matrix **mat, matrix *from, int row_offset, int col_offset,
                                int rows, int cols, double **p_data)
{
    double **from_data = from->data;
    if (rows <= 950)
    {
        for (int i = 0; i < rows; i++)
        {
            p_data[i] = from_data[row_offset + i] + col_offset;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < (rows / 4) * 4; i += 4)
        {
            p_data[i] = from_data[row_offset + i] + col_offset;
            p_data[i + 1] = from_data[row_offset + i + 1] + col_offset;
            p_data[i + 2] = from_data[row_offset + i + 2] + col_offset;
            p_data[i + 3] = from_data[row_offset + i + 3] + col_offset;
        }

        for (int i = (rows / 4) * 4; i < rows; i++)
        {
            p_data[i] = from_data[row_offset + i] + col_offset;
        }
    }

    // this matrix references FROM's data
    (from->ref_cnt)++;

    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->tp = from->tp;
    (*mat)->data = p_data;
    (*mat)->is_1d = (rows == 1 || cols == 1);
    (*mat)->parent = from;
    (*mat)->ref_cnt = 1;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols)
{
    if (!from)
    {
        fprintf(stderr, "ERROR: INVALID: value `to` and `from` for rows: %d\n");
        return -1;
    }
    else if (rows <= 0)
    {
        fprintf(stderr, "ERROR: INVALID: value for rows: %d\n", rows);
        return -1;
    }
    else if (cols <= 0)
    {
        fprintf(stderr, "ERROR: INVALID: value for cols: %d\n", cols);
        return -1;
    }
    else if ((row_offset + rows) > from->rows)
    {
        fprintf(stderr, "Too many rows in matrix slice; tries to make a "
                        "matrix slice of [%d:%d] in a matrix with only %d rows.\n",
                row_offset, row_offset + rows, from->rows);
        return -1;
    }
    else if ((col_offset + cols) > from->cols)
    {
        fprintf(stderr, "Too many cols in matrix slice; tries to make a "
                        "matrix slice of [%d:%d] in a matrix with only %d cols.\n",
                col_offset, row_offset + cols, from->cols);
        return -1;
    }

    *mat = malloc(sizeof(matrix));
    if (!(*mat))
    {
        fprintf(stderr, "ERROR: the allocation for matrix failed.\n");
        return -1;
    }

    double **p_data = malloc(rows * sizeof(double *));
    if (!p_data)
    {
        fprintf(stderr, "ERROR: the allocation for matrix failed.\n");
        return -1;
    }

    allocate_matrix_ref_helper(mat, from, row_offset, col_offset, rows, cols, p_data);

    return 0;
}

// HELPER
void allocate_matrix_uninitialized_helper(matrix **mat, int rows, int cols)
{
    *mat = malloc(sizeof(matrix));
    if (!mat)
    {
        fprintf(stderr, "ERROR: the allocation for matrix failed.\n");
        return -1;
    }

    double **p_data = malloc(rows * sizeof(double *));
    if (!p_data)
    {
        fprintf(stderr, "ERROR: the allocation for matrix failed.\n");
        return -1;
    }

    // uses malloc instead of calloc
    int mat_size = rows * cols;
    double *data = malloc(mat_size * sizeof(double));
    if (!data)
    {
        fprintf(stderr, "ERROR: the allocation for matrix failed.\n");
        return -1;
    }
    double *tp = malloc(mat_size * sizeof(double));
    if (!tp)
    {
        fprintf(stderr, "ERROR: the allocation for matrix failed.\n");
        return -1;
    }
#pragma omp parallel for if (rows > 500)
    for (int i = 0; i < rows; i++)
    {
        p_data[i] = data + (i * cols);
    }

    (*mat)->data = p_data;
    (*mat)->tp = tp;
    (*mat)->cols = cols;
    (*mat)->rows = rows;
    (*mat)->is_1d = (rows == 1 || cols == 1);
    (*mat)->parent = NULL;

    (*mat)->ref_cnt = 1;
}

/*
 * Like allocate_matrix but the data is not set to 0.
 */
int allocate_matrix_uninitialized(matrix **mat, int rows, int cols)
{
    if (cols <= 0)
    {
        fprintf(stderr, "Bad value for cols: %d\n", cols);
        return -1;
    }
    else if (rows <= 0)
    {
        fprintf(stderr, "Bad value for rows: %d\n", rows);
        return -1;
    }

    allocate_matrix_uninitialized_helper(mat, rows, cols);

    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat)
{
    if (!mat)
        return;
    if (mat->parent)
    {
        (mat->parent->ref_cnt)--;
        if (mat->parent->ref_cnt < 1)
            deallocate_matrix(mat->parent);
    }
    else if ((mat->ref_cnt <= 1))
    {
        free(mat->data[0]);
        free(mat->data);
        free(mat->tp);
        free(mat);
        return;
    }

    if (mat->ref_cnt <= 1)
    {
        free(mat->data);
        free(mat);
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col)
{
    if (!mat)
    {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val)
{
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val)
{
    const __m256d val256 = _mm256_set1_pd(val);

#pragma omp parallel for if (mat->rows > 500)
    for (int i = 0; i < mat->rows; i++)
    {
        double *row = mat->data[i];

        for (int j = 0; j < (mat->cols / 16) * 16; j += 16)
        {
            _mm256_storeu_pd(row + j, val256);
            _mm256_storeu_pd(row + j + 4, val256);
            _mm256_storeu_pd(row + j + 8, val256);
            _mm256_storeu_pd(row + j + 12, val256);
        }

        for (int j = (mat->cols / 16) * 16; j < mat->cols; j++)
        {
            row[j] = val;
        }
    }
}
void add_matrix_helper(matrix *result, matrix *mat1, matrix *mat2, int num_rows, int num_cols)
{
    if (num_rows >= 1000)
    {
        if (num_cols >= 4000)
        {
            int inner_loop_len = (num_cols / 16) * 16;
#pragma omp parallel for
            for (int i = 0; i < num_rows; i++)
            {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                add_inner_loop(result, row1, row2, dest, inner_loop_len);
                add_outer_loop(result, row1, row2, dest, inner_loop_len, num_cols);
            }
        }
        else if (num_cols >= 1000)
        {
            int inner_loop_len = (num_cols / 4) * 4;
#pragma omp parallel for
            for (int i = 0; i < num_rows; i++)
            {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                add_inner_loop(result, row1, row2, dest, inner_loop_len);
                add_regular_loop(result, row1, row2, dest, num_cols);
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < num_rows; i++)
            {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                add_regular_loop(result, row1, row2, dest, num_cols);
            }
        }
    }
    else
    {
        for (int i = 0; i < num_rows; i++)
        {
            double *row1 = mat1->data[i];
            double *row2 = mat2->data[i];
            double *dest = result->data[i];
            add_regular_loop(result, row1, row2, dest, num_cols);
        }
    }
}

void add_inner_loop(matrix *result, double *row1, double *row2, double *dest, int inner_loop_len)
{
    int k0Block = 0;
    int k4Block = 4;
    int k8Block = 8;
    int k12Block = 12;
    for (int j = 0; j < inner_loop_len; j += 16)
    {
        _mm256_storeu_pd(
            dest + j,      
            _mm256_add_pd( 
                _mm256_loadu_pd(row1 + j + k0Block),
                _mm256_loadu_pd(row2 + j + k0Block)));
        _mm256_storeu_pd(
            dest + j + 4,  
            _mm256_add_pd( 
                _mm256_loadu_pd(row1 + j + k4Block),
                _mm256_loadu_pd(row2 + j + k4Block)));
        _mm256_storeu_pd(
            dest + j + 8,  
            _mm256_add_pd( 
                _mm256_loadu_pd(row1 + j + k8Block),
                _mm256_loadu_pd(row2 + j + k8Block)));
        _mm256_storeu_pd(
            dest + j + 12, 
            _mm256_add_pd( 
                _mm256_loadu_pd(row1 + j + k12Block),
                _mm256_loadu_pd(row2 + j + k12Block)));
    }
}

void add_outer_loop(matrix *result, double *row1, double *row2, double *dest, int inner_loop_len, int num_cols)
{
    for (int j = inner_loop_len; j < num_cols; j++)
    {
        dest[j] = row1[j] + row2[j];
    }
}

void add_regular_loop(matrix *result, double *row1, double *row2, double *dest, int num_cols)
{
    for (int j = 0; j < num_cols; j++)
    {
        dest[j] = row1[j] + row2[j];
    }
}


/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    if (!(result && mat1 && mat2))
    {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }

      int mat1_cols = mat1->cols;
    int mat2_cols = mat2->cols;
    int mat1_rows = mat1->rows;
    int mat2_rows = mat2->rows;
    int result_cols = result->cols;
    int result_rows = result->rows;

    if (mat1_cols != mat2_cols)
    {
        fprintf(stderr, "Error: The number of columns in mat1 (%d) does not match the number of columns in mat2 (%d).\n", mat1_cols, mat2_cols);
        return -1;
    }
    if (mat1_rows != mat2_rows)
    {
        fprintf(stderr, "Error: The number of rows in mat1 (%d) does not match the number of rows in mat2 (%d).\n", mat1_rows, mat2_rows);
        return -1;
    }
    if (result_cols != mat1_cols)
    {
        fprintf(stderr, "Error: The number of columns in the result matrix (%d) does not match the number of columns in mat1 (%d).\n", result_cols, mat1_cols);
        return -1;
    }
    if (result_rows != mat1_rows)
    {
        fprintf(stderr, "Error: The number of rows in the result matrix (%d) does not match the number of rows in mat1 (%d).\n", result_rows, mat1_rows);
        return -1;
    }

    int num_rows = result->rows;
    int num_cols = result->cols;

    add_matrix_helper(result, mat1, mat2, num_rows, num_cols);

    return 0;
}

void sub_matrix_helper(matrix *result, matrix *mat1, matrix *mat2)
{
    int num_rows = result->rows;
    int num_cols = result->cols;

    if (num_rows >= 1000)
    {
        if (num_cols >= 4000)
        {
            int inner_loop_len = (num_cols / 16) * 16;
#pragma omp parallel for
            for (int i = 0; i < num_rows; i++)
            {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < inner_loop_len; j += 16)
                {
                    _mm256_storeu_pd(
                        dest + j,      
                        _mm256_sub_pd( 
                            _mm256_loadu_pd(row1 + j),
                            _mm256_loadu_pd(row2 + j)));
                    _mm256_storeu_pd(
                        dest + j + 4,  
                        _mm256_sub_pd( 
                            _mm256_loadu_pd(row1 + j + 4),
                            _mm256_loadu_pd(row2 + j + 4)));
                    _mm256_storeu_pd(
                        dest + j + 8,  
                        _mm256_sub_pd( 
                            _mm256_loadu_pd(row1 + j + 8),
                            _mm256_loadu_pd(row2 + j + 8)));
                    _mm256_storeu_pd(
                        dest + j + 12, 
                        _mm256_sub_pd( 
                            _mm256_loadu_pd(row1 + j + 12),
                            _mm256_loadu_pd(row2 + j + 12)));
                }

                for (int j = inner_loop_len; j < num_cols; j++)
                {
                    dest[j] = row1[j] - row2[j];
                }
            }
        }
        else if (num_cols >= 1000)
        {
            int inner_loop_len = (num_cols / 4) * 4;
#pragma omp parallel for
            for (int i = 0; i < num_rows; i++)
            {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < inner_loop_len; j += 4)
                {
                    _mm256_storeu_pd(
                        dest + j,      // where to store result
                        _mm256_sub_pd( // sub mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j),
                            _mm256_loadu_pd(row2 + j)));
                }

                for (int j = inner_loop_len; j < num_cols; j++)
                {
                    dest[j] = row1[j] - row2[j];
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < num_rows; i++)
            {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < num_cols; j++)
                {
                    dest[j] = row1[j] - row2[j];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < num_rows; i++)
        {
            double *row1 = mat1->data[i];
            double *row2 = mat2->data[i];
            double *dest = result->data[i];
            for (int j = 0; j < num_cols; j++)
            {
                dest[j] = row1[j] - row2[j];
            }
        }
    }
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    if (!(result && mat1 && mat2))
    {
        fprintf(stderr, "ERROR: Given a null pointer\n");
        return -1;
    }

    if (mat1->rows != result->rows || mat2->rows != result->rows)
    {
        fprintf(stderr, "ERROR:  Row values don't match for these matrices: `%d`, `%d`, `%d`\n", result->rows, mat1->rows, mat2->rows);
        return -1;
    }
    else if (mat1->cols != result->cols || mat2->cols != result->cols)
    {
        fprintf(stderr, "ERROR:  Col values don't match for these matrices: `%d`, `%d`, `%d`\n", result->cols, mat1->cols, mat2->cols);
        return -1;
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */

void transpose(matrix *mat)
{
    double *tp = mat->tp;
    int rows = mat->rows;
    int cols = mat->cols;

    const int DIV = 4;
    const int NUM_THREADS = DIV * DIV;

    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int rm = id / DIV;
        int cm = id % DIV;
        int r_start = (rows / DIV) * rm;
        int c_start = (cols / DIV) * cm;
        int r_stop = r_start + rows / DIV + (rows % DIV) * (rm == DIV - 1);
        int c_stop = c_start + cols / DIV + (cols % DIV) * (cm == DIV - 1);
        for (int i = r_start; i < r_stop; i++)
        {
            double *row = mat->data[i];
            for (int j = c_start; j < c_stop; j++)
            {
                tp[rows * j + i] = row[j];
            }
        }
    }
}

void simple_mul_helper(matrix *result, matrix *mat1, matrix *mat2)
{
#pragma omp parallel for if (mat1->rows > 100 && mat2->cols > 100)
    for (int i = 0; i < mat1->rows; i++)
    {
        for (int k = 0; k < mat1->cols; k++)
        {
            for (int j = 0; j < mat2->cols; j++)
            {
                result->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
            }
        }
    }
}

void simple_mul(matrix *result, matrix *mat1, matrix *mat2)
{
    for (int i = 0; i < mat1->rows; i++)
        for (int j = 0; j < mat2->cols; j++)
            result->data[i][j] = 0;

    simple_mul_helper(result, mat1, mat2);
}

int mul_matrix(matrix* output, matrix* inputMatrix1, matrix* inputMatrix2) {
    if (!(output && inputMatrix1 && inputMatrix2)) {
        fprintf(stderr, "Error: Given a null pointer\n");
        return -1;
    }

    if (inputMatrix1->cols != inputMatrix2->rows) {
        fprintf(stderr, "Error: The number of columns in matrix1 does not match the number of rows in matrix2: %d, %d\n", inputMatrix1->cols, inputMatrix2->rows);
        return -1;
    }
    else if (output->rows != inputMatrix1->rows) {
        fprintf(stderr, "Error: The number of rows in the output matrix does not match the number of rows in matrix1: %d, %d\n", output->rows, inputMatrix1->rows);
        return -1;
    }
    else if (output->cols != inputMatrix2->cols) {
        fprintf(stderr, "Error: The number of columns in the output matrix does not match the number of columns in matrix2: %d, %d\n", output->cols, inputMatrix2->cols);
        return -1;
    }

    if (inputMatrix1->rows < 200 && inputMatrix2->cols < 200) {
        simple_mul(output, inputMatrix1, inputMatrix2);
        return 0;
    }

    int numRows1 = inputMatrix1->rows;
    int numCols2 = inputMatrix2->cols;
    int numCols1 = inputMatrix1->cols;

    transpose(inputMatrix2);
    double* transposedPtr = inputMatrix2->tp;

    const int BLOCK = 8;
    const int BLOCK4 = 4 * BLOCK;

    const int DIV = 4;
    const int NUM_THREADS = DIV * DIV;

    omp_set_num_threads(NUM_THREADS);

    const __m256d ZERO = _mm256_set1_pd(0);
#pragma omp parallel
    {
        int threadID = omp_get_thread_num();
        int rowMultiplier = threadID / DIV;
        int colMultiplier = threadID % DIV;

        int rowStart = (numRows1 / DIV) * rowMultiplier;
        int colStart = (numCols2 / DIV) * colMultiplier;

        int rowStop = rowStart + numRows1 / DIV + (numRows1 % DIV) * (rowMultiplier == DIV - 1);
        int colStop = colStart + numCols2 / DIV + (numCols2 % DIV) * (colMultiplier == DIV - 1);

        const int colLoop = colStart + ((colStop - colStart) / 4) * 4;

        for (int i = rowStart; i < rowStop; i += BLOCK) {
            for (int j = colStart; j < colLoop; j += BLOCK4) {
                int rowStopCondition = (rowStop < (i + BLOCK)) ? rowStop : (i + BLOCK);
                for (int r = i; r < rowStopCondition; r++) {
                    double* row = inputMatrix1->data[r];
                    double* res = output->data[r];
                    int colStopCondition = (colStop < (j + BLOCK4)) ? colLoop : (j + BLOCK4);
                    for (int c = j; c < colStopCondition; c += 4) {
                        double* col1 = transposedPtr + numCols1 * c;
                        double* col2 = transposedPtr + numCols1 * (c + 1);
                        double* col3 = transposedPtr + numCols1 * (c + 2);
                        double* col4 = transposedPtr + numCols1 * (c + 3);
                        __m256d rowSegment, columnSegment, dotProduct4 = ZERO;
                        for (int k = 0; k < numCols1; k++) {
                            rowSegment = _mm256_broadcast_sd(row + k);
                            columnSegment = _mm256_set_pd(col4[k], col3[k], col2[k], col1[k]);
                            dotProduct4 = _mm256_fmadd_pd(rowSegment, columnSegment, dotProduct4);
                        }
                        _mm256_storeu_pd(res + c, dotProduct4);
                    }
                }
            }
            for (int j = colLoop; j < colStop; j += BLOCK) {
                int rowStopCondition = (rowStop < (i + BLOCK)) ? rowStop : (i + BLOCK);
                for (int r = i; r < rowStopCondition; r++) {
                    double* row = inputMatrix1->data[r];
                    double* res = output->data[r];
                    int colStopCondition = (colStop < (j + BLOCK)) ? colStop : (j + BLOCK);
                    for (int c = j; c < colStopCondition; c++) {
                        double* col = transposedPtr + numCols1 * c;
                        double dotProduct = 0;
                        for (int k = 0; k < numCols1; k++) {
                            dotProduct += row[k] * col[k];
                        }
                        res[c] = dotProduct;
                    }
                }
            }
        }
    }
    return 0;
}


void pow_matrix_pow_helper(int pow, matrix *temp, matrix *prev, matrix *swap, matrix *r1, matrix *r2)
{
    while (pow > 1)
    {
        mul_matrix(temp, prev, prev);
        swap = prev;
        prev = temp;
        temp = swap;

        pow >>= 1;

        if (pow & 1)
        {
            mul_matrix(r1, r2, prev);
            swap = r2;
            r2 = r1;
            r1 = swap;
        }
    }
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */

int pow_matrix(matrix *result, matrix *mat, int pow)
{
    if (!(result && mat))
    {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }

    if (mat->rows != mat->cols)
    {
        fprintf(stderr, "can't exponentiate nonsquare matrix. Rows: `%d`, cols: `%d`\n", mat->rows, mat->cols);
        return -1;
    }
    else if (result->rows != result->cols)
    {
        fprintf(stderr, "Bad result matrix dimensions. Rows: `%d`, cols: `%d`\n", result->rows, result->cols);
        return -1;
    }
    else if (mat->rows != result->rows)
    {
        fprintf(stderr, "Given matrix isn't compatible with result. Input size: `%d`, output size: `%d`\n", mat->rows, result->rows);
        return -1;
    }
    else if (pow < 0)
    {
        fprintf(stderr, "can't exponentiate matrix to negative power: `%d`\n", pow);
        return -1;
    }

    matrix *prev = NULL;
    if (allocate_matrix_uninitialized(&prev, mat->rows, mat->rows))
    {
        fprintf(stderr, "Failed to alloc matrix\n");
        return -1;
    }
    copy_matrix(prev, mat);

    matrix *temp = NULL;
    if (allocate_matrix(&temp, mat->rows, mat->rows))
    {
        fprintf(stderr, "Failed to alloc matrix\n");
        return -1;
    }
#pragma omp parallel for if (mat->rows > 500)
    for (int i = 0; i < mat->rows; i++)
    {
        temp->data[i][i] = 1;
    }

    copy_matrix(result, temp);

    matrix *swap;

    matrix *r1 = NULL;
    if (allocate_matrix_uninitialized(&r1, mat->rows, mat->rows))
    {
        fprintf(stderr, "Failed to alloc matrix\n");
        return -1;
    }

    matrix *r2 = result;

    if (pow & 1)
    {
        copy_matrix(result, prev);
    }

    pow_matrix_pow_helper(pow, temp, prev, swap, r1, r2);

    if (result != r1)
    {
        deallocate_matrix(r1);
    }
    else
    {
        copy_matrix(result, r2);
        deallocate_matrix(r2);
    }

    deallocate_matrix(prev);
    deallocate_matrix(temp);
    return 0;
}

void sq_matrix_helper(matrix *result, matrix *mat, int dot_len)
{
    for (int i = 0; i < result->rows; i++)
    {
        for (int j = 0; j < result->cols; j++)
        {
            int dot_product = 0;
            for (int k = 0; k < dot_len; k++)
            {
                dot_product += mat->data[i][k] * mat->data[k][j];
            }
            result->data[i][j] = dot_product;
        }
    }
}

/*
 * Square a matrix and store the result.
 * Has no checks since it's always called from pow_matrix, which already
 * has performed all nec checks
 */
void sq_matrix(matrix *result, matrix *mat)
{
    int dot_len = mat->cols;
    sq_matrix_helper(result, mat, dot_len);
}

void store_double(double* dest, double* source, int start, int end) {
    for (int i = start; i < end; i++) {
        dest[i] = source[i];
    }
}

void store_double_256(double* dest, double* source, int start, int end) {
    for (int i = start; i < end; i += 4) {
        _mm256_storeu_pd(dest + i, _mm256_loadu_pd(source + i));
    }
}

int copy_matrix(matrix* result, matrix* mat) {
    if (!(result && mat)) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    if (mat->rows != result->rows) {
        fprintf(stderr, "Row values don't match. Result: `%d`, input: `%d`\n", result->rows, mat->rows);
        return -1;
    }
    else if (mat->cols != result->cols) {
        fprintf(stderr, "Col values don't match. Result: `%d`, input: `%d`\n", result->cols, mat->cols);
        return -1;
    }

    if (result->cols > 1000) {
        int inner_loop_len = (result->cols / 16) * 16;

        #pragma omp parallel for if (result->rows > 500)
        for (int i = 0; i < result->rows; i++) {
            double* source = mat->data[i];
            double* dest = result->data[i];

            store_double_256(dest, source, 0, inner_loop_len);
            store_double(dest, source, inner_loop_len, result->cols);
        }
    }
    else if (result->cols > 100) {
        int inner_loop_len = (result->cols / 4) * 4;

        #pragma omp parallel for if (result->rows > 500)
        for (int i = 0; i < result->rows; i++) {
            double* source = mat->data[i];
            double* dest = result->data[i];

            store_double_256(dest, source, 0, inner_loop_len);
            store_double(dest, source, inner_loop_len, result->cols);
        }
    }
    else {
        #pragma omp parallel for if (result->rows > 500)
        for (int i = 0; i < result->rows; i++) {
            double* source = mat->data[i];
            double* dest = result->data[i];
            store_double(dest, source, 0, result->cols);
        }
    }

    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    if (!(result && mat)) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    if (mat->rows != result->rows) {
        fprintf(stderr, "Row values don't match. Result: `%d`, input: `%d`\n", result->rows, mat->rows);
        return -1;
    }
    else if (mat->cols != result->cols) {
        fprintf(stderr, "Col values don't match. Result: `%d`, input: `%d`\n", result->cols, mat->cols);
        return -1;
    }

    const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));

    int num_rows = result->rows;
    int num_cols = result->cols;

    if (num_rows > 500) {
        if (num_cols > 2000) {
            int inner_loop_len = (num_cols / 16) * 16;
            parallel_neg_block(result, mat, inner_loop_len, mask);
        }
        else if (num_cols > 500) {
            int inner_loop_len = (num_cols / 4) * 4;
            parallel_neg_block(result, mat, inner_loop_len, mask);
        }
        else {
            neg_block(result, mat, num_cols);
        }
    }
    else {
        if (num_cols > 2000) {
            int inner_loop_len = (num_cols / 16) * 16;
            neg_block(result, mat, inner_loop_len);
        }
        else if (num_cols > 500) {
            int inner_loop_len = (num_cols / 4) * 4;
            neg_block(result, mat, inner_loop_len);
        }
        else {
            neg_block(result, mat, num_cols);
        }
    }

    return 0;
}

void neg_block(matrix *result, matrix *mat, int inner_loop_len) {
    #pragma omp parallel for
    for (int i = 0; i < result->rows; i++) {
        double *source = mat->data[i];
        double *dest = result->data[i];

        for (int j = 0; j < inner_loop_len; j++) {
            dest[j] = -1 * source[j];
        }
    }
}

void parallel_neg_block(matrix *result, matrix *mat, int inner_loop_len, const __m256d mask) {
    #pragma omp parallel for
    for (int i = 0; i < result->rows; i++) {
        double *source = mat->data[i];
        double *dest = result->data[i];

        for (int j = 0; j < inner_loop_len; j += 16) {
            _mm256_storeu_pd(
                dest + j,
                _mm256_xor_pd(
                    _mm256_loadu_pd(source + j),
                    mask));
            _mm256_storeu_pd(
                dest + j + 4,
                _mm256_xor_pd(
                    _mm256_loadu_pd(source + j + 4),
                    mask));
            _mm256_storeu_pd(
                dest + j + 8,
                _mm256_xor_pd(
                    _mm256_loadu_pd(source + j + 8),
                    mask));
            _mm256_storeu_pd(
                dest + j + 12,
                _mm256_xor_pd(
                    _mm256_loadu_pd(source + j + 12),
                    mask));
        }

        for (int j = inner_loop_len; j < result->cols; j++) {
            dest[j] = -1 * source[j];
        }
    }
}

void abs_matrix_helper(matrix *result, matrix *mat, __m256d mask, int num_rows, int num_cols)
{
#pragma omp parallel for if (num_rows > 500)
    for (int i = 0; i < result->rows; i++)
    {
        if (num_cols <= 500)
        {
            double *src = mat->data[i];
            double *dst = result->data[i];
            for (int j = 0; j < result->cols; j++)
            {
                dst[j] = fabs(src[j]);
            }
        }
        else
        {
            int inner_loop_len = (num_cols/4) * 4;

            double *src = mat->data[i];
            double *dst = result->data[i];

            for (int j = 0; j < inner_loop_len; j += 4)
            {
                _mm256_storeu_pd(
                    dst + j,     
                    _mm256_and_pd(
                        _mm256_loadu_pd(src + j),
                        mask));
            }

            for (int j = inner_loop_len; j < result->cols; j++)
            {
                dst[j] = fabs(src[j]);
            }
        }
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat)
{
    if (!(result && mat && 1))
    {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    if (mat->cols != result->cols)
    {
        fprintf(stderr, "Column values don't match. Result: `%d`, input: `%d`\n", result->cols, mat->cols);
        return -1;
    }
    else if (mat->rows != result->rows)
    {
        fprintf(stderr, "Row values don't match. Result: `%d`, input: `%d`\n", result->rows, mat->rows);
        return -1;
    }

    const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    int numRows = result->rows;
    int numCols = result->cols;

    abs_matrix_helper(result, mat, mask, numRows, numCols);

    return 0;
}
