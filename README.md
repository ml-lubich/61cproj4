# numc

Woohoo we are Justin and Kirill and we have allegedly made matrices fast in C.

What we did in project 4:
## Task 1 and 2
- Task 1 and Task 2 were pretty trivial.
  - Task 1 was just following the function definitions
    - We're both pretty comfortable with C syntax so this was easy
    - Task 2 was a little tricky but mostly was just reading documentation
## Task 3
- Lots of reading
- Lots of helper functions to make the subscript functions less of a pain/code spaghetti than it already was
- We created a `slice` struct for storing values of slices and passing its information to the necessary functions 
- Same thing with the `mat_idx` struct, except it stores values for either an int or a slice
  - `to_mat_idx()` is a function that takes in either a slice or an int and converts it into a `mat_idx` struct for further processing
    - Saves a lot of repetitive calls
    - If it receives a slice, it saves infomation that is gotten from `PySlice_GetIndicesEx` into the `mat_idx` struct
  - `parse_key()` parses the keys (either an int or a slice) for the subscript functions
    - calls `to_mat_idx()` and then calls errors as appropriate if the processed keys are bad
    - otherwise it returns the int/slice info via the `slice` struct to the subscript functions
- For `subscript`:
  - After calling `parse_key()` on the key, we know exactly what parts of the matrix to expose/display, with information contained within the `slice` struct
    - The rest of the function is calling errors as appropriate
  - For `set_subscript`:
    - Basically the same as `subscript` except we have to call the appropriate setting functions
      - Can go into a for loop to set multiple locations, if inputted an array
- For `get_value`
  - We call `PyArgs_ParseTuple` to process the arguments, confirm they're row/col values, and that they're valid
  - Then we get the value from the matrix woo
- Similar thing with `set_value`, except we have an extra argument, the value to set the matrix cell to
- For `Matrix61c_as_number`, we map the Python fields to the appropriate functions in `matrix.c`
- For `abs` and `neg`, we don't really need type checking since the only argument is already guaranteed to be a matrix
- The rest of the matrix operand functions mainly involve parsing with `PyObject_TypeCheck` and then calling the appropriate C functions
  - We made extensive use of the provided `get_shape` helper function to set the shape of the matrices
- Of course we always allocated a new matrix
  - How wasteful
## Task 4
- The first thing we did were the extremely basic optimizations
  - Originally, all the functions used the `get` and `set` functions to access parts of a matrix
    - It's a nice abstraction, but it's an unnecessary function call
    - We replaced it with directly accessing the matrix data
  - Loop unrolling in `fill`, `add`, `sub`, etc
- SIMD was a pretty easy idea to wrap your head around
  - Grab from the matrix data starting at an address and progress in steps of 4
- We did stuff like `double *dest = result->data[i];`, storing pointers to regularly accessed areas on the stack
  - Maybe the compiler would optimize for us, but you never know
- At first we indiscriminantly added the `#pragma` parallelization directives for the for loops
  - Very naive idea
  - Added a lot of overhead for the smaller matrices
    - We tried adding huge manual if/else blocks to account for matrix shapes and had huge amounts of unnecessarily repeated code
      - Eventually we discovered that `#pragma`s have an if statement
- We did not play around a lot with the positioning of the parallel loops; we mostly parallelized the outer loops
- For `neg`, we took advantage of the fact that all the data was stored in doubles
  - To negate a double, you just need to flip the sign bit
  - We have a `__m256d` mask containing doubles with only the sign bit set, and then we XOR it with the values in data to negate them
- For `abs`, we use a similar masking idea
  - Absolute value of a double involves turning the sign bit off
  - This time the mask has all bits set to 1 except the sign bit, which is 0
  - We AND it with the data in order to get their proper absolute value
  - Got stuck for a time on `abs` vs `fabs` >:(
- We originally had data be a bunch of separated arrays, but we coalesced them into one giant double, so it only needed a single `calloc` call to be properly set
  - We parallelized the setting of the data rows after the underlying array was allocated
- We created a new allocation function that initialized data with `malloc` instead of zero-setting them
  - Slightly less overhead and also useful for some matrices in `pow`
- For `pow`, we allocate several new matrices to hold data in, since it has to make several calls to `mul`, and the result matrix must be different from the inputs
  - Sometimes not zero allocated
  - We initialize the result matrix to be the identity matrix
  - If the power input is nonzero, we go into a while loop, otherwise the identity is returned as desired
  - In the while loop:
    - We take advantage of the fact that `x^n = x^{2 * n/2}` if even and `x*x^{2 * n/2}` otherwise
    - We use 2 temporary matrix pointers that each swap between the current matrix value, and the next matrix value after being squared
      - Less overhead than copying between matrices all the time
      - We square with a specialized matrix squaring function without matrix checks, since we always guarantee matrix dimension safety
        - Only called from the power function
    - We floor divide the power by 2 `(n >>= 1)`
    - If the power is odd, we multiply the result by the square that we have calculated
      - Basically we replicate: `A^13 = A^0b1101 = 1*A^8 + 1*A^4 + 0*A^2 + 1*A^1`
      - The result matrix is also swapped between 2 pointers to reduce copying overhead
    - Then we deallocate the allocated matrices and we're done
 - For mul, we used parallelization, simd, tiling, and the transpose.
      -We achieved parallelization by giving each thread a chunk of the matrix (each chunk being independent of course)
      - simd was done through vectorization
       - tiling was done for less cache misses, we did tiling for both rows and columns
        - the biggest optimization was taking the transpose of the second matrix, we now had only 1 cache miss per column, this worked well with simd vectorization
# 61cproj4
