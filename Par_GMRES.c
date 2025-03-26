#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
/*
 I have added parallelization to the inner loops in the Arnoldi iterations.
 */
//Forward Declarations
double vector_norm(const double* x, int n);
double dot_product(const double* x, const double* y, int n);
void mat_vec_mult(double** A, const double* x, double* result, int n);
typedef struct {
    double** H;
    double** Q;
} Arnoldi; //want to return two matrices so create a struct to package both these
typedef struct{
    double* res; //residual norm history over iterations
    double* x; //vector containing estimated solution at each step
    int num_iters; //number of iterations
} GMRES_res;
Arnoldi* arnoldi(double** A, double* u, int m, int ulen);
//------------------HELPER FUNCTIONS-----------------------------------------------
//Computes dot product for vector multiplication
//Parallel version of the dot product: divide the loop iterations evenly amongst the threads. We avoid race conditions which could occur if threads tried to update a shared variable concurrently using "reduction(+:sum)" as each thread maintains their own copy of the partial sum, and then we compute the total sum at the end by adding these all up
double dot_product(const double* x, const double* y, int n) {
    double sum = 0.0;
    int i;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

// ============================
//
// Computes matrix-vector product: result = A * x.
// A assumed to be a square matrix of dimension n x n.
// ============================
//parallel version of matrix-vector multiplication, we parallelize the outer loop over i and give each thread a private copy of the loop variable j (so the threads don't conflict when running over the same iteration of the inner loop). We use static scheduling of the work as each iteration of the loop takes roughly the same amount of work so iterations are divided evenly among threads
void mat_vec_mult(double** A, const double* x, double* result, int n) {
    int i, j;
    #pragma omp parallel for private(j) schedule(static)
    for (i = 0; i < n; i++) {
        double temp = 0.0;
        for (j = 0; j < n; j++) {
            temp += A[i][j] * x[j];
        }
        result[i] = temp; //each thread has their own copy of this to avoid race conditions that may occur if they tried to update a shared variable concurrently
    }

    // Multiply each row of A with vector x.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
}
//---Vector Norm Helper function--------------
double vector_norm(const double* x, int n) {
    return sqrt(dot_product(x, x, n));
}
//------Build our tridiagonal matrix for specified n------------------
double** build_tridiagonal_matrix(int n){
    double** A = calloc(n, sizeof(double*));
    for(int j = 0; j < n; j++){
        A[j] = calloc(n, sizeof(double)); //initialise rows
    }
    for(int i = 0; i<n; i++){
        A[i][i] = -4.0; //set main diagonal to -4
        if(i>0){
            A[i][i-1]=1.0; //set the diagonal to the left of main diagonal to 1
        }
        if(i<n-1){
            A[i][i+1]=1.0; //set diagonal to the right of main diagonal to 1
        }
    }
    return A;
}
//---------Build vector, each consecutive entry is equally spaced apart, final value is 1--------
double* build_vec(int n){
    double* b = malloc(n*sizeof(double));
    for(int i = 0; i<n; i++){
        b[i] = (double)(i+1)/(double)n; //cast numerator and denominator for proper divisionnrather than rounding
    }
    return b;
}

//------------ARNOLDI FUNCTION-------------------------------------------------------
Arnoldi* arnoldi(double** A, double* u, int m, int ulen) {
    double two_norm_u = 0.0;
    for (int i = 0; i < ulen; i++){
        two_norm_u += u[i] * u[i];
    }
    two_norm_u = sqrt(two_norm_u);

    Arnoldi* a = calloc(1, sizeof(Arnoldi));
    if (!a) {
        perror("FAILED TO CALLOC ARNOLDI STRUCT");
        exit(EXIT_FAILURE);
    }
    a->Q = calloc(m+1, sizeof(double*));
    a->H = calloc(m+1, sizeof(double*));
    for (int i = 0; i < m+1; i++){
        a->Q[i] = calloc(ulen, sizeof(double));
        if (!a->Q[i]) {
            fprintf(stderr, "FOR Q WE FAILED TO CALLOC ROW %d: ", i);
            perror("");
            exit(EXIT_FAILURE);
        }
    }
    for (int j = 0; j < m+1; j++){
        a->H[j] = calloc(m, sizeof(double));
        if (!a->H[j]) {
            fprintf(stderr, "FOR H WE FAILED TO CALLOC ROW %d: ", j);
            perror("");
            exit(EXIT_FAILURE);
        }
    }
    
    // Set the first basis vector Q[0] = u/||u||
    for (int i = 0; i < ulen; i++){
        a->Q[0][i] = u[i] / two_norm_u;
    }
    
    // Arnoldi iteration
    for (int j = 0; j < m; j++){
        double* v = calloc(ulen, sizeof(double));
        if (!v) {
            fprintf(stderr, "FAILED TO CALLOC FOR V AT ITERATION %d : ", j);
            perror("");
            exit(EXIT_FAILURE);
        }
        // Compute v = A * Q[j]
        mat_vec_mult(A, a->Q[j], v, ulen);
        
        for (int i = 0; i <= j; i++){
            a->H[i][j] = dot_product(a->Q[i], v, ulen);
            // In Orthogonalizing v against all previous Q vectors, we parallelize the subtraction over the vector elements. Each iteration is independent as each v[q] is updated using the qth element of the ith basis vector and a scalar, so doesn't depend on other iterations' values.
            #pragma omp parallel for schedule(static)
            for (int q = 0; q < ulen; q++){
                v[q] -= a->H[i][j] * a->Q[i][q];
            }
        }
        
        double two_norm_v = 0.0;
        // Parallelize the norm computation with reduction: each thread gets a partial sum stored in a private copy of two_norm_v and we add these partial sums together at the end
        #pragma omp parallel for reduction(+:two_norm_v) schedule(static)
        for (int z = 0; z < ulen; z++){
            two_norm_v += v[z] * v[z];
        }
        two_norm_v = sqrt(two_norm_v);
        a->H[j+1][j] = two_norm_v;
        if (two_norm_v < 1e-10) {
            free(v);
            break;  // Krylov subspace has been fully spanned
        }
        // Normalize v to form the next basis vector Q[j+1], done in parallel as each element can be computed independently.
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < ulen; r++){
            a->Q[j+1][r] = v[r] / two_norm_v;
        }
        free(v);
    }
    return a;
}
//===============LEAST SQUARES==================================
//This allows us to minimize ||Beta*e_{1} - H_{k}y||_{2} in y to update our initial guess at each iteration. We use Gaussian elimination, assuming that H^{T}_{k} * H_{k} is non-singular. We use the normal equations : H^{T}_{k}Beta*e_{1} = H^{T}_{k}H_{k}*y. Assuming k not too large allows for Gaussian elimination.

double least_squares(double **H, int k, double beta, double* y) {
    int i, j, r; //for our for loops
    double** HTH = malloc(k*sizeof(double*)); //malloc for H^{T}*H
    //now initialise the row entries of H^{T}*H to zero
    for (i = 0; i < k; i++) {
        HTH[i] = calloc(k, sizeof(double));
    }
    // Allocate vector: H^{T}_{k}Beta*e_{1} of length k
    double* HTB = calloc(k, sizeof(double));
    // Note: H_ is the first (k+1) rows and k columns of H
    // Since e1 = [1, 0, ..., 0]^T, H_{k}^T*(beta*e1) = beta * (first row of H_{k})
    for (i = 0; i < k; i++) {
        HTB[i] = beta * H[0][i];
    }
    // Form HTH = H_{k}^T * H_{k}.
    for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
            double sum = 0.0;
            for (r = 0; r < k+1; r++) { // r = 0,...,k
                sum += H[r][i] * H[r][j];
            }
            HTH[i][j] = sum;
        }
    }
    // Solve HTH y = HTBe_{1} w/ Gaussian elimination
    for (i = 0; i < k; i++) {
        // Pivoting
        double pivot = HTH[i][i];
        if (fabs(pivot) < 1e-12) {
            fprintf(stderr, "Pivot too small.\n");
            exit(EXIT_FAILURE);
        }
        // Normalize row i
        for (j = i; j < k; j++) {
            HTH[i][j] /= pivot;
        }
        HTB[i] /= pivot;
        // Eliminate below
        for (int p = i+1; p < k; p++) {
            double factor = HTH[p][i];
            for (j = i; j < k; j++) {
                HTH[p][j] -= factor * HTH[i][j];
            }
            HTB[p] -= factor * HTB[i];
        }
    }
    // Back substitution.
    for (i = k-1; i >= 0; i--) {
        double sum = 0.0;
        for (j = i+1; j < k; j++) {
            sum += HTH[i][j] * y[j];
        }
        y[i] = HTB[i] - sum;
    }
    // Compute residual norm: r = beta*e1 - H_{k}*y.
    double res_sq = 0.0;
    for (i = 0; i < k+1; i++) {
        double temp = 0.0;
        for (j = 0; j < k; j++) {
            temp += H[i][j] * y[j];
        }
        double diff = (i == 0 ? beta - temp : -temp);
        res_sq += diff * diff;
    }
    double res_norm = sqrt(res_sq);
    // Free temporary arrays.
    for (i = 0; i < k; i++) {
        free(HTH[i]);
    }
    free(HTH);
    free(HTB);
    return res_norm;
}
//======GMRES FUNCTION=============================
//A is the matrix in Ax = b, b is the rhs vector, m is number of iterations, n is the size of A (nxn) and b (nx1), tol is our desired tolerance for stopping the algo
GMRES_res* gmres(double** A, double* b, int m, int n, double tol){
    double* x_0 = calloc(n, sizeof(double)); //initial guess (zero vector)
    double* r_0 = malloc(n*sizeof(double)); //inital residual = b
    double beta = vector_norm(b,n); //initial beta, the vector norm of r_0
    double* res = malloc(m*sizeof(double)); //the history of the residuals over m iterations stored here
    double* y_temp = malloc(m*sizeof(double)); //temporarily store y at each iteration here
    Arnoldi* arn = arnoldi(A, b, m, n); //do m Arnoldi iterations
    int count = 0; //count the number of iterations performed by the loop in case we terminate early with the stopping rule being satisfied
    for(int j = 1; j<=m; j++){
        count++;
        double res_nrm = least_squares(arn->H, j, beta, y_temp); //solve the least squares problem of Beta*e_1 - H_{k}*y, this y is y_temp
        res[j-1] = res_nrm; //update residual history
        if(res_nrm < tol){
            break; //exit if stopping rule satisfied
        }
    }
    double* y = malloc(count*sizeof(double)); //this is where we store our final y value
    least_squares(arn->H, count, beta, y);
    // Compute s = V_{k}*y. Here, V_{k} are the first k Arnoldi basis vectors.
        double* s = calloc(n, sizeof(double));
        for (int j = 0; j < count; j++) {
            for (int i = 0; i < n; i++) {
                s[i] += y[j] * arn->Q[j][i];
            }
        }
    double* final_x = s; //our initial guess was x_0 = 0 therefore our final guess is just s = V_{k}y
    GMRES_res* gmres_res = malloc(sizeof(GMRES_res));
    gmres_res -> res = res;
    gmres_res -> x = s;
    gmres_res -> num_iters = count;
    //free stuff up
    free(y_temp);
    free(y);
    for(int i = 0; i<m+1; i++){
        free(arn->Q[i]);
        free(arn->H[i]);
    }
    free(arn->Q);
    free(arn->H);
    free(arn);
    free(x_0);
    return gmres_res;
}
int main(void){
    int sizes[] = {8, 16, 32, 64, 126, 256};
    int numsizes = sizeof(sizes)/sizeof(sizes[0]); // 6 sizes
    double tol = 1e-10; // tolerance for GMRES

    for (int k = 0; k < numsizes; k++){
        int n = sizes[k];       // current matrix size
        int m = n / 2;          // GMRES iterations = n/2
        double* b = build_vec(n);
        double** A = build_tridiagonal_matrix(n);
        GMRES_res* result = gmres(A, b, m, n, tol);
        double b_norm = vector_norm(b, n); // norm of b

        // Print final iteration's ratio:
        double final_ratio = result->res[result->num_iters - 1] / b_norm;
        printf("n = %d, m = %d, Final iteration (%d): ||r_k||/||b|| = %e\n", n, m, result->num_iters, final_ratio);

        // Free matrix A
        for (int i = 0; i < n; i++){
            free(A[i]);
        }
        free(A);
        free(b);
        free(result->res);
        free(result->x);
        free(result);
    }
    return 0;
}

