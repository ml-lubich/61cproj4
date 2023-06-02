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
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
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
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows <= 0) {
        fprintf(stderr, "Bad value for rows: %d\n", rows);
        return -1;
    } else if (cols <= 0) {
        fprintf(stderr, "Bad value for cols: %d\n", cols);
        return -1;
    }

    *mat = malloc(sizeof(matrix));
    if (!mat) {
        fprintf(stderr, "Malloc for matrix failed.\n");
        return -1;
    }

    double **p_data = malloc(rows*sizeof(double *));
    if (!p_data) {
        fprintf(stderr, "Malloc for matrix rows failed.\n");
        return -1;
    }

    // calloc zero initializes rows
    int mat_size = rows *cols;
    double *data = calloc(mat_size, sizeof(double));
    if (!data) {
        fprintf(stderr, "Calloc for matrix data failed.\n");
        return -1;
    }
    double *tp = malloc(mat_size * sizeof(double));
    if (!tp) {
        fprintf(stderr, "Malloc for matrix transpose failed.\n");
        return -1;
    }
#pragma omp parallel for if (rows > 500)
    for (int i = 0; i < rows; i++) {
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

    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {    
    if (!from) {
        fprintf(stderr, "Bad reference to `from` matrix\n");
        return -1;
    } else if (rows <= 0) {
        fprintf(stderr, "Bad value for rows: %d\n", rows);
        return -1;
    } else if (cols <= 0) {
        fprintf(stderr, "Bad value for cols: %d\n", cols);
        return -1;
    } else if ((row_offset + rows) > from->rows) {
        fprintf(stderr, "Too many rows in matrix slice; tries to make a "
                        "matrix slice of [%d:%d] in a matrix with only %d rows.\n", 
                        row_offset, row_offset+rows, from->rows);
        return -1;
    } else if ((col_offset + cols) > from->cols) {
        fprintf(stderr, "Too many cols in matrix slice; tries to make a "
                        "matrix slice of [%d:%d] in a matrix with only %d cols.\n", 
                        col_offset, row_offset+cols, from->cols);
        return -1;
    }

    *mat = malloc(sizeof(matrix));
    if (!(*mat)) {
        fprintf(stderr, "Malloc for matrix failed.\n");
        return -1;
    }

    double **p_data = malloc(rows * sizeof(double *));
    if (!p_data) {
        fprintf(stderr, "Malloc for matrix data rows failed.\n");
        return -1;
    }
    
    double **from_data = from->data;
    if (rows > 950) {
        # pragma omp parallel for
        for (int i = 0; i < (rows / 4) * 4; i += 4) {
            p_data[i] = from_data[row_offset + i] + col_offset;
            p_data[i + 1] = from_data[row_offset + i + 1] + col_offset;
            p_data[i + 2] = from_data[row_offset + i + 2] + col_offset;
            p_data[i + 3] = from_data[row_offset + i + 3] + col_offset;
        }

        for (int i = (rows / 4) * 4; i < rows; i++) {
            p_data[i] = from_data[row_offset + i] + col_offset;
        }
    } else {
        for (int i = 0; i < rows; i++) {
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


    return 0;
}

/*
 * Like allocate_matrix but the data is not set to 0.
 */
int allocate_matrix_uninitialized(matrix **mat, int rows, int cols) {
    if (rows <= 0) {
        fprintf(stderr, "Bad value for rows: %d\n", rows);
        return -1;
    } else if (cols <= 0) {
        fprintf(stderr, "Bad value for cols: %d\n", cols);
        return -1;
    }

    *mat = malloc(sizeof(matrix));
    if (!mat) {
        fprintf(stderr, "Malloc for matrix failed.\n");
        return -1;
    }

    double **p_data = malloc(rows*sizeof(double *));
    if (!p_data) {
        fprintf(stderr, "Malloc for matrix rows failed.\n");
        return -1;
    }

    // uses malloc instead of calloc
    int mat_size = rows *cols;
    double *data = malloc(mat_size * sizeof(double));
    if (!data) {
        fprintf(stderr, "Malloc for matrix data failed.\n");
        return -1;
    }
    double *tp = malloc(mat_size * sizeof(double));
    if (!tp) {
        fprintf(stderr, "Malloc for matrix transpose failed.\n");
        return -1;
    }
    #pragma omp parallel for if (rows > 500)
    for (int i = 0; i < rows; i++) {
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

    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    if (!mat) return;
    if (mat->parent) {
        (mat->parent->ref_cnt)--;
        if (mat->parent->ref_cnt < 1) 
	        deallocate_matrix(mat->parent);
    } else if ((mat->ref_cnt <= 1)) {
        free(mat->data[0]);
        // # pragma omp parallel for
        // for (int i = 0; i < mat->rows; i++) {
        // }

        // repeat next if block
        free(mat->data);
        free(mat->tp);
        free(mat);
        return;
    }
    
    if (mat->ref_cnt <= 1) {
        free(mat->data);
        free(mat);
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    if (!mat) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    const __m256d val256 = _mm256_set1_pd(val);

    #pragma omp parallel for if(mat->rows > 500)
    for (int i = 0; i < mat->rows; i++) {
        double *row = mat->data[i];

        for (int j = 0; j < (mat->cols / 16) * 16; j += 16) {
            _mm256_storeu_pd(row + j, val256);
            _mm256_storeu_pd(row + j + 4, val256);
            _mm256_storeu_pd(row + j + 8, val256);
            _mm256_storeu_pd(row + j + 12, val256);
        }

        for (int j = (mat->cols / 16) * 16; j < mat->cols; j++) {
            row[j] = val;
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (!(result && mat1 && mat2)) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }

    if (mat1->rows != result->rows || mat2->rows != result->rows) {
        fprintf(stderr, "Row values don't match for these matrices: `%d`, `%d`, `%d`\n", result->rows, mat1->rows, mat2->rows);
        return -1;
    } else if (mat1->cols != result->cols || mat2->cols != result->cols) {
        fprintf(stderr, "Col values don't match for these matrices: `%d`, `%d`, `%d`\n", result->cols, mat1->cols, mat2->cols);
        return -1;
    }

    int num_rows = result->rows;
    int num_cols = result->cols;

    if (num_rows >= 1000) {
        if (num_cols >= 4000) {
            int inner_loop_len = (num_cols / 16) * 16;
            # pragma omp parallel for
            for (int i = 0; i < num_rows; i++) {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < inner_loop_len; j += 16) {
                    _mm256_storeu_pd(
                        dest + j, // where to store result
                        _mm256_add_pd( // add mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j),
                            _mm256_loadu_pd(row2 + j)
                        )
                    );
                    _mm256_storeu_pd(
                        dest + j + 4, // where to store result
                        _mm256_add_pd( // add mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j + 4),
                            _mm256_loadu_pd(row2 + j + 4)
                        )
                    );
                    _mm256_storeu_pd(
                        dest + j + 8, // where to store result
                        _mm256_add_pd( // add mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j + 8),
                            _mm256_loadu_pd(row2 + j + 8)
                        )
                    );
                    _mm256_storeu_pd(
                        dest + j + 12, // where to store result
                        _mm256_add_pd( // add mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j + 12),
                            _mm256_loadu_pd(row2 + j + 12)
                        )
                    );

                    // result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
                }

                for (int j = inner_loop_len; j < num_cols; j++) {
                    dest[j] = row1[j] + row2[j];
                }
            }

        } else if (num_cols >= 1000) {
            int inner_loop_len = (num_cols / 4) * 4;
            # pragma omp parallel for
            for (int i = 0; i < num_rows; i++) {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < inner_loop_len; j += 4) {
                    _mm256_storeu_pd(
                        dest + j, // where to store result
                        _mm256_add_pd( // add mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j),
                            _mm256_loadu_pd(row2 + j)
                        )
                    );

                    // result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
                }

                for (int j = inner_loop_len; j < num_cols; j++) {
                    dest[j] = row1[j] + row2[j];
                }
            }
        } else {
            # pragma omp parallel for
            for (int i = 0; i < num_rows; i++) {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < num_cols; j++) {
                    dest[j] = row1[j] + row2[j];
                }
            }
        }


    }
    else
    {
        for (int i = 0; i < num_rows; i++) {
            double *row1 = mat1->data[i];
            double *row2 = mat2->data[i];
            double *dest = result->data[i];
            for (int j = 0; j < num_cols; j++) {
                dest[j] = row1[j] + row2[j];
            }
        }
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (!(result && mat1 && mat2)) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }

    if (mat1->rows != result->rows || mat2->rows != result->rows) {
        fprintf(stderr, "Row values don't match for these matrices: `%d`, `%d`, `%d`\n", result->rows, mat1->rows, mat2->rows);
        return -1;
    } else if (mat1->cols != result->cols || mat2->cols != result->cols) {
        fprintf(stderr, "Col values don't match for these matrices: `%d`, `%d`, `%d`\n", result->cols, mat1->cols, mat2->cols);
        return -1;
    }

    int num_rows = result->rows;
    int num_cols = result->cols;

    if (num_rows >= 1000) {
        if (num_cols >= 4000) {
            int inner_loop_len = (num_cols / 16) * 16;
            # pragma omp parallel for
            for (int i = 0; i < num_rows; i++) {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < inner_loop_len; j += 16) {
                    _mm256_storeu_pd(
                        dest + j, // where to store result
                        _mm256_sub_pd( // sub mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j),
                            _mm256_loadu_pd(row2 + j)
                        )
                    );
                    _mm256_storeu_pd(
                        dest + j + 4, // where to store result
                        _mm256_sub_pd( // sub mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j + 4),
                            _mm256_loadu_pd(row2 + j + 4)
                        )
                    );
                    _mm256_storeu_pd(
                        dest + j + 8, // where to store result
                        _mm256_sub_pd( // sub mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j + 8),
                            _mm256_loadu_pd(row2 + j + 8)
                        )
                    );
                    _mm256_storeu_pd(
                        dest + j + 12, // where to store result
                        _mm256_sub_pd( // sub mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j + 12),
                            _mm256_loadu_pd(row2 + j + 12)
                        )
                    );
                }

                for (int j = inner_loop_len; j < num_cols; j++) {
                    dest[j] = row1[j] - row2[j];
                }
            }

        } else if (num_cols >= 1000) {
            int inner_loop_len = (num_cols / 4) * 4;
            # pragma omp parallel for
            for (int i = 0; i < num_rows; i++) {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < inner_loop_len; j += 4) {
                    _mm256_storeu_pd(
                        dest + j, // where to store result
                        _mm256_sub_pd( // sub mat1 and mat2's rows
                            _mm256_loadu_pd(row1 + j),
                            _mm256_loadu_pd(row2 + j)
                        )
                    );
                }

                for (int j = inner_loop_len; j < num_cols; j++) {
                    dest[j] = row1[j] - row2[j];
                }
            }
        } else {
            # pragma omp parallel for
            for (int i = 0; i < num_rows; i++) {
                double *row1 = mat1->data[i];
                double *row2 = mat2->data[i];
                double *dest = result->data[i];
                for (int j = 0; j < num_cols; j++) {
                    dest[j] = row1[j] - row2[j];
                }
            }
        }


    }
    else
    {
        for (int i = 0; i < num_rows; i++) {
            double *row1 = mat1->data[i];
            double *row2 = mat2->data[i];
            double *dest = result->data[i];
            for (int j = 0; j < num_cols; j++) {
                dest[j] = row1[j] - row2[j];
            }
        }
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */

void transpose(matrix *mat) {
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
        for (int i = r_start; i < r_stop; i++) {
            double *row = mat->data[i];
            for (int j = c_start; j < c_stop; j++) {
                tp[rows * j + i] = row[j];
            }
        }
    }
}

void simple_mul(matrix *result, matrix *mat1, matrix *mat2) {
    for (int i = 0; i < mat1->rows; i++)
        for (int j = 0; j < mat2->cols; j++)
            result->data[i][j] = 0;
#pragma omp parallel for if (mat1->rows > 100 && mat2->cols > 100)
    for (int i = 0; i < mat1->rows; i++) {
        for (int k = 0; k < mat1->cols; k++) {
            for (int j = 0; j < mat2->cols; j++) {
                result->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
            }
        }
    }
}

int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Check for null pointers
    if (!(result && mat1 && mat2)) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    
    // Check matrix dimensions
    if (mat1->cols != mat2->rows) {
        fprintf(stderr, "mat1's cols don't match mat2's rows: `%d`, `%d`\n", mat1->cols, mat2->rows);
        return -1;
    } else if (result->rows != mat1->rows) {
        fprintf(stderr, "result's rows don't match mat1's rows: `%d`, `%d`\n", result->rows, mat1->rows);
        return -1;
    } else if (result->cols != mat2->cols) {
        fprintf(stderr, "result's cols don't match mat2's cols: `%d`, `%d`\n", result->cols, mat2->cols);
        return -1;
    }

    // Check if the matrices are small for simple multiplication
    if (mat1->rows < 200 && mat2->cols < 200) {
        simple_multiply(result, mat1, mat2);
        return 0;
    }

    int mat1Rows = mat1->rows;
    int mat2Cols = mat2->cols;
    int n = mat1->cols;

    // Transpose mat2 for better memory access pattern
    transpose(mat2);
    double *transposeMat2 = mat2->transpose;

    // Set loop parameters and number of threads
    const int BLOCK = 8;
    const int BLOCK4 = 4 * BLOCK;
    const int DIV = 4;
    const int NUM_THREADS = DIV * DIV;

    // Set the number of threads for OpenMP
    omp_set_num_threads(NUM_THREADS);

    // Create a constant vector of zeros
    const __m256d ZERO = _mm256_set1_pd(0);

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        int rowBlock = threadId / DIV;
        int colBlock = threadId % DIV;

        int rowStart = (mat1Rows / DIV) * rowBlock;
        int colStart = (mat2Cols / DIV) * colBlock;

        int rowStop = rowStart + mat1Rows / DIV + (mat1Rows % DIV) * (rowBlock == DIV - 1);
        int colStop = colStart + mat2Cols / DIV + (mat2Cols % DIV) * (colBlock == DIV - 1);

        const int colLoop = colStart + ((colStop - colStart) / 4) * 4;

        for (int i = rowStart; i < rowStop; i += BLOCK) {
            for (int j = colStart; j < colLoop; j += BLOCK4) {
                int rowStopCond = (rowStop < (i + BLOCK)) ? rowStop : (i + BLOCK);
                for(int r = i; r < rowStopCond; r++) {
                    double *row = mat1->data[r];
                    double *res = result->data[r];
                    int colStopCond = (colStop < (j + BLOCK4)) ? colLoop : (j + BLOCK4);
                    for (int c = j; c < colStopCond; c += 4) {
                        double *col1 = transposeMat2 + n * c;
                        double *col2 = transposeMat2 + n * (c + 1);
                        double *col3 = transposeMat2 + n * (c + 2);
                        double *col4 = transposeMat2 + n * (c + 3);
                        __m256d rs, c4, dp4 = ZERO;
                        for (int k = 0; k < n; k++) {
                            rs = _mm256_broadcast_sd(row + k);
                            c4 = _mm256_set_pd(col4[k], col3[k], col2[k], col1[k]);
                            dp4 = _mm256_fmadd_pd(rs, c4, dp4);
                        }
                        _mm256_storeu_pd(res + c, dp4);
                    }
                }
            }
            for (int j = colLoop; j < colStop; j += BLOCK) {
                int rowStopCond = (rowStop < (i + BLOCK)) ? rowStop : (i + BLOCK);
                for (int r = i; r < rowStopCond; r++) {
                    double *row = mat1->data[r];
                    double *res = result->data[r];
                    int colStopCond = (colStop < (j + BLOCK)) ? colStop : (j + BLOCK);
                    for (int c = j; c < colStopCond; c++) {
                        double *col = transposeMat2 + n * c;
                        double dp = 0;
                        for (int k = 0; k < n; k++) {
                            dp += row[k] * col[k];
                        }
                        res[c] = dp;
                    }
                }
            }
        }
    }
    return 0;
    }


/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */

int pow_matrix(matrix *result, matrix *mat, int pow)
{
    if (!(result && mat)) {
        fprintf(stderr, "Error: Given a null pointer\n");
        return -1;
    }
    
    if (mat->rows != mat->cols) {
        fprintf(stderr, "Error: Cannot exponentiate nonsquare matrix. Rows: `%d`, Cols: `%d`\n", mat->rows, mat->cols);
        return -1;
    } else if (result->rows != result->cols) {
        fprintf(stderr, "Error: Invalid result matrix dimensions. Rows: `%d`, Cols: `%d`\n", result->rows, result->cols);
        return -1;
    } else if (mat->rows != result->rows) {
        fprintf(stderr, "Error: Incompatible matrices. Input size: `%d`, Output size: `%d`\n", mat->rows, result->rows);
        return -1;
    } else if (pow < 0) {
        fprintf(stderr, "Error: Cannot exponentiate matrix to negative power: `%d`\n", pow);
        return -1;
    }

    matrix *prev = NULL;
    if (allocate_matrix_uninitialized(&prev, mat->rows, mat->rows)) {
        fprintf(stderr, "Error: Failed to allocate matrix\n");
        return -1;
    }
    copy_matrix(prev, mat);

    matrix *temp = NULL;
    if (allocate_matrix(&temp, mat->rows, mat->rows)) {
        fprintf(stderr, "Error: Failed to allocate matrix\n");
        return -1;
    }

    #pragma omp parallel for if (mat->rows > 450)
    for (int i = 0; i < mat->rows; i++) {
        temp->data[i][i] = 1;
    }

    copy_matrix(result, temp);

    matrix *swap;
    matrix *r1 = NULL;
    if (allocate_matrix_uninitialized(&r1, mat->rows, mat->rows)) {
        fprintf(stderr, "Error: Failed to allocate matrix\n");
        return -1;
    }

    matrix *r2 = result;

    if (pow & 1) {
        copy_matrix(result, prev);
    }

    while (pow > 1) {
        mul_matrix(temp, prev, prev);
        swap = prev;
        prev = temp;
        temp = swap;

        pow >>= 1;

        if (pow & 1) {
            mul_matrix(r1, r2, prev);
            swap = r2;
            r2 = r1;
            r1 = swap;
        }
    }

    if (result == r1) {
        copy_matrix(result, r2);
        deallocate_matrix(r2);
    } else {
        deallocate_matrix(r1);
    }

    deallocate_matrix(prev);
    deallocate_matrix(temp);
    return 0;
}

/* 
 * Square a matrix and store the result. 
 * Has no checks since it's always called from pow_matrix, which already
 * has performed all nec checks
 */
void sq_matrix(matrix *result, matrix *mat) {
    int dot_len = mat->cols;
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            int dot_product = 0;
            for (int k = 0; k < dot_len; k++) {
                dot_product += mat->data[i][k] * mat->data[k][j];
            }
            result->data[i][j] = dot_product;
        }
    }
}

/*
 * Copy MAT's values into RESULT.
 * Return 0 upon success, nonzero value upon failure;
 */
int copy_matrix(matrix *result, matrix *mat) {
    if (!(result && mat)) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    if (mat->rows != result->rows) {
        fprintf(stderr, "Row values don't match. Result: `%d`, input: `%d`\n", result->rows, mat->rows);
        return -1;
    } else if (mat->cols != result->cols) {
        fprintf(stderr, "Col values don't match. Result: `%d`, input: `%d`\n", result->cols, mat->cols);
        return -1;
    }

    if (result->cols > 1000) {
        int inner_loop_len = (result->cols / 16) * 16;

        # pragma omp parallel for if(result->rows>500)
        for (int i = 0; i < result->rows; i++) {
            double *source = mat->data[i];
            double *dest = result->data[i];

            for (int j = 0; j < inner_loop_len; j += 16) {
                _mm256_storeu_pd(dest + j, _mm256_loadu_pd(source + j));
                _mm256_storeu_pd(dest + j + 4, _mm256_loadu_pd(source + j + 4));
                _mm256_storeu_pd(dest + j + 8, _mm256_loadu_pd(source + j + 8));
                _mm256_storeu_pd(dest + j + 12, _mm256_loadu_pd(source + j + 12));
            }

            for (int j = inner_loop_len; j < result->cols; j++) {
                dest[j] = source[j];
            }
        }
    } else if (result->cols > 100) {
        int inner_loop_len = (result->cols / 4) * 4;

        # pragma omp parallel for if(result->rows>500)
        for (int i = 0; i < result->rows; i++) {
            double *source = mat->data[i];
            double *dest = result->data[i];

            for (int j = 0; j < inner_loop_len; j += 4) {
                _mm256_storeu_pd(dest + j, _mm256_loadu_pd(source + j));
            }

            for (int j = inner_loop_len; j < result->cols; j++) {
                dest[j] = source[j];
            }
        }
    } else {
        # pragma omp parallel for if(result->rows>500)
        for (int i = 0; i < result->rows; i++) {
            double *source = mat->data[i];
            double *dest = result->data[i];
            for (int j = 0; j < result->cols; j++) {
                dest[j] = source[j];
            }
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
    } else if (mat->cols != result->cols) {
        fprintf(stderr, "Col values don't match. Result: `%d`, input: `%d`\n", result->cols, mat->cols);
        return -1;
    }

    const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));

    int num_rows = result->rows;
    int num_cols = result->cols;

    if (num_rows > 500) {
        if (num_cols > 2000) {
            neg_matrix_large(result, mat, mask);
        } else if (num_cols > 500) {
            neg_matrix_medium(result, mat, mask);
        } else {
            neg_matrix_small(result, mat);
        }
    } else {
        if (num_cols > 2000) {
            neg_matrix_large(result, mat, mask);
        } else if (num_cols > 500) {
            neg_matrix_medium(result, mat, mask);
        } else {
            neg_matrix_small(result, mat);
        }
    }

    return 0;
}

void neg_matrix_large(matrix *result, matrix *mat, const __m256d mask) {
    int inner_loop_len = (result->cols / 16) * 16;

    #pragma omp parallel for
    for (int i = 0; i < result->rows; i++) {
        double *source = mat->data[i];
        double *dest = result->data[i];

        for (int j = 0; j < inner_loop_len; j += 16) {
            neg_block(dest + j, source + j, mask);
            neg_block(dest + j + 4, source + j + 4, mask);
            neg_block(dest + j + 8, source + j + 8, mask);
            neg_block(dest + j + 12, source + j + 12, mask);
        }

        for (int j = inner_loop_len; j < result->cols; j++) {
            dest[j] = -1 * source[j];
        }
    }
}

void neg_matrix_medium(matrix *result, matrix *mat, const __m256d mask) {
    int inner_loop_len = (result->cols / 4) * 4;

    #pragma omp parallel for
    for (int i = 0; i < result->rows; i++) {
        double *source = mat->data[i];
        double *dest = result->data[i];

        for (int j = 0; j < inner_loop_len; j += 4) {
            neg_block(dest + j, source + j, mask);
        }

        for (int j = inner_loop_len; j < result->cols; j++) {
            dest[j] = -1 * source[j];
        }
    }
}

void neg_matrix_small(matrix *result, matrix *mat) {
    #pragma omp parallel for
    for (int i = 0; i < result->rows; i++) {
        double *source = mat->data[i];
        double *dest = result->data[i];

        for (int j = 0; j < result->cols; j++) {
            dest[j] = -1 * source[j];
        }
    }
}

void neg_block(double *dest, double *source, const __m256d mask) {
    _mm256_storeu_pd(
        dest,
        _mm256_xor_pd(
            _mm256_loadu_pd(source),
            mask
        )
    );
}

/*
 * Store the result of taking the$ absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if (!(result && mat)) {
        fprintf(stderr, "Given a null pointer\n");
        return -1;
    }
    if (mat->rows != result->rows) {
        fprintf(stderr, "Row values don't match. Result: `%d`, input: `%d`\n", result->rows, mat->rows);
        return -1;
    } else if (mat->cols != result->cols) {
        fprintf(stderr, "Col values don't match. Result: `%d`, input: `%d`\n", result->cols, mat->cols);
        return -1;
    }

    const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    int num_rows = result->rows;
    int num_cols = result->cols;

    #pragma omp parallel for if (num_rows > 500)
    for (int i = 0; i < result->rows; i++) {
        if (num_cols > 500) {
            int inner_loop_len = (num_cols / 4) * 4;

            double *source = mat->data[i];
            double *dest = result->data[i];

            for (int j = 0; j < inner_loop_len; j += 4) {
                __m256d source_vec = _mm256_loadu_pd(source + j);
                __m256d abs_vec = _mm256_and_pd(source_vec, mask);
                _mm256_storeu_pd(dest + j, abs_vec);
            }

            for (int j = inner_loop_len; j < result->cols; j++) {
                dest[j] = fabs(source[j]);
            }
        } else {
            double *source = mat->data[i];
            double *dest = result->data[i];
            for (int j = 0; j < result->cols; j++) {
                dest[j] = fabs(source[j]);
            }
        }
    }
    return 0;
}

