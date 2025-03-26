#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/*=====================================================================
 GMRES is useful for solving Ax = b
 Generalized Minimal Residual Algorithm
 The GMRES is an algo that at step q computes the best least squares solution for the Krylov subspace of order q for A and b.
 Recall that the residual r is the l2 norm of (b - Ax):
 
 r = ||b - Ax||^{2}_{2}
 
 If b is an element of the column space of A => there exists a solution with residual = 0
 If b is not an element of the column space of A => there is no solution with zero residual, and our best case is a Least Squares solution.
 
 Since inversion of A can be computationally expensive, the Weak Cayley-Hamilton theorem gives us an insight into the usefulness of Krylov subspaces in the inversion of A:
 
 If A is invertible, and we have the Krylov subspace representation of A wrt b:
 b, Ab, A^{2}b, ... , A^{n}b,...
 The first n+1 of these can be rewritten as:
 w_{0}b + w_{1}Ab + w_{2}A^{2}b + ... + w_{n}A^{n}b
 
 for some non-zero w_{i}'s.
 Furthermore, if k<=n is the first integer (smallest) such that w_{k} != 0, then the Weak Cayley-Hamilton Theorem says A^{-1}b can be represented as:
 
 A^{-1}b = 1/w_{k}[w_{k+1}b + w_{k+2}Ab + ... + w_{n}A^{n-k-1}b]
 
 So A^{-1} can be represented as matrix vector products => Krylov subspace a fitting choice for this representation.
 
 Steps for solving Ax = b with GMRES:
 
 1. Choose x_{0} and get the residual b - Ax_{0}
 2. For chosen steps or until threshold satisfied:
 Do the Arnoldi iteration
 3. Get approx. solution:
 x_{k} = x_{0} + s_{k} where s_{k} = V_{k}p_{k}. Here, V_{k} is the matrix whose columns represent the orthonormal basis vectors for the Krylov subspace generated during the Arnoldi iteration.  p_{k} minimizes:
 
 Z(y) = ||Ke_{1} - H_{k}y|| <- this is the l_{2} norm, where K is the norm of the first residual r_{0} and e_{1} is the first column of the (k+1)x(k+1) identity matrix. With Arnoldi, after you perform k iterations you get an upper hessenberg matrix of size (k+1) x k, which we denote as H_{k} in this case.
 
    x is approximately x_{0} + V_{k}y, so we correct our initial guess for the solution for Ax = b at each step with V_{k}y.
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
double dot_product(const double* x, const double* y, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

// ============================
//
// Computes matrix-vector product: result = A * x.
// A assumed to be a square matrix of dimension n x n.
// ============================
void mat_vec_mult(double** A, const double* x, double* result, int n) {
    // Zero initialize the result vector using a loop.
    for (int i = 0; i < n; i++) {
        result[i] = 0.0;
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
Arnoldi* arnoldi(double** A, double* u, int m, int ulen){
    //ulen is the length of the vector u
    double two_norm_u = 0.0;
    for(int i = 0; i<ulen; i++){
        two_norm_u += (u[i]*u[i]);
    }
    two_norm_u = sqrt(two_norm_u);
    //allocate memeory for Q (orthonormal basis vectors) and H (upper hessenberg matrix)
    //set them to zero w/ calloc
    Arnoldi* a = calloc(1, sizeof(Arnoldi)); //allocate memory for 1 arnoldi structure
    if(!a){
        perror("FAILED TO CALLOC ARNOLDI STRUCT");
        exit(EXIT_FAILURE);
    }
    a->Q = calloc(m+1, sizeof(double*));
    a->H = calloc(m+1, sizeof(double*));
    for(int i = 0; i<m+1; i++){
        a->Q[i] = calloc(ulen, sizeof(double));
        if(a->Q[i]==NULL){
            fprintf(stderr, "FOR Q WE FAILED TO CALLOC ROW %d: ", i);
            perror("");
            exit(EXIT_FAILURE);
        }
    }
    for(int j = 0; j<m+1; j++){
        a->H[j] = calloc(m, sizeof(double));
        if(a->H[j]==NULL){
            fprintf(stderr, "FOR H WE FAILED TO CALLOC ROW %d: ", j);
            perror("");
            exit(EXIT_FAILURE);
        }
    }
    for(int i = 0; i<ulen; i++){
        a->Q[0][i] = (u[i]/two_norm_u); //set the first element of Q (row wise storing of the basis vectors) to the normalised u.
    }
    for(int j = 0; j < m; j++){
        double* v = calloc(ulen, sizeof(double));
        if(!v){
            fprintf(stderr, "FAILED TO CALLOC FOR V AT ITERATION %d : ", j);
            perror("");
            exit(EXIT_FAILURE);
        }
        mat_vec_mult(A, a->Q[j], v, ulen); //mat_vec_mult(matrix, vec, where to store result, dimension of multiplication)
        //Next, we orthogonalise v against all previous basis vectors ensuring our basis is orthogonal at all stages
        for(int i = 0; i<=j; i++){
            a->H[i][j] = dot_product(a->Q[i], v, ulen);
            for(int q = 0; q<ulen; q++){
                v[q] -= a->H[i][j]*(a->Q[i][q]);
            }
        }
        double two_norm_v = 0.0;
        for(int z = 0; z<ulen; z++){
            two_norm_v += (v[z]*v[z]);
        }
        two_norm_v = sqrt(two_norm_v);
        a->H[j+1][j] = two_norm_v;
        if(two_norm_v < 1e-10){
            free(v);
            break;    //if v is very small we have spanned the Krylov subspace
        }
        for(int r = 0; r<ulen; r++){
            a->Q[j+1][r] = ((v[r])/(two_norm_v));
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
