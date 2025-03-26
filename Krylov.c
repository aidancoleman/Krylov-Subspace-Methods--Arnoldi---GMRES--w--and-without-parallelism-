#include <stdio.h>
#include <math.h>
#include <stdlib.h>
/*With Arnoldi, we want to find the approximate eigenvalues and eigenvectors of a large matrix A by finding an orthonormal basis for the Krylov subspace of the matrix.
More on Krylov subspaces:
 for a matrix A and starting vector b, the kth Krylov Subspace is:
 span(b, Ab, A^2b, ... , A^(k-1)b)
 
 The intuitive idea behind this Krylov subspace construction is to capture the action of A on b in a lower-dimensional space. These matrix vector multiplications are relatively computationally cheap, and they help for A large and sparse (or when we don't explicitly have A but have a way to build it) in solving linear systems with A of the form Ax=b, they also allow us to examine the eigenvalues/vectors (approximating them).
 
 Hessenberg matrices: these are matrices that are "almost" triangular, upper hessenberg means almost upper triangular and lower hessenberg means almost lower triangular.
 Tridiagonal matrices are special type that are both upper and lower hessenberg
 
 Upper Hessenberg:
 A is called upper Hessenberg if a_{ij} = 0 for all i > j+1, e.g. row values to the left and below of j+1 are 0Allow for entries one above the diagonal.
 
 Lower Hessenberg: A is called lower Hessenberg if a_{ij} = 0 for all j > i+1, e.g column values to the right and above of i+1 are 0. Allow for entries 1 below the diagonal.
 
 Arnoldi reduces A to a hessenberg form allowing us to approximate eigenvalues.
 
 Steps:
 
 Compute A = QHQ^T, where H is upper hessenberg and Q is unitary => QQ^T = 1.
 Dimension  of A (nxn) can be massive, so instead we focus on the first m << n columns of the factorization:
            AQ = QH
 So on left hand side only need the first m columns of Q_{m}, so Q_{m} \in \mathbb{C}^{n x m}
 And only need the first m columns of the hessenberg matrix H. More specifically, we need the (m+1)xm upper left section of H, because due to its hessenberg structure, H only interacts with the first m+1 columns of Q => we therefore have:
        AQ_{m} = Q_{m+1}Htilda, where we denote Htilda as the (m+1)xm upper-left section of H
 
Representing this multiplication element wise:
        Aq_{m} = q_{1}h_{1, m} + q_{2}h_{2, m} + ... + q_{m+1}h_{m+1, m}
        => q_{m+1} = (Aq_{m} - h_{1, m}q_{1} - h_{2, m}q_{2} - ... - h_{m, m}q_{m})/h_{m+1, m}
 
 Arnoldi is Gram-Schmidt method that constructs the h_{i, j} and orthonormal vectors q_{i, j}
 
 Pseudocode:
 
 1. choose b, then set q_{1} = b/||b||_{2} //choose q_{1} to be the normalised version of b
 2. for m = 1,2,.. do:
    v = Aq_{m}
    for j = 1, 2, 3,... do:
        h_{j, m} = q^{T}_{j}v //orthogonalise v against all previous basis vectors j<=m
        update v = v - h_{j, m}q_{j} //orthogonal vector
        **end inner for loop**
    h_{m+1, m} = ||v||_{2}
    q_{m+1} = v/h_{m+1, m} //this is our orthonormal vector after normalising v
 **end outer for loop**
end
 
we need to also evaluate Aq_{m} and perform some vector operations at each iteration
 The q_{j} form an orthonormal basis of the successive Krylov subspaces.
 
 These Krylov subspaces are expected to provide useful info regarding the dominant eigenvalues and eigenvectors of A on a lower-dimensional scale
 i.e. span(b, Ab, A^{2}b, ... , A^{m-1}b) = span(q_1, q_2, ... , q_m)
 
 We now need to find the eigenvalues from the Arnoldi iteration.
 Let H_m = Q^{T}_{m}AQ_{m}
 be the original Htilda with the last row removed.
 
 AT EACH STEP WE COMPUTE THE EIGENVALUES OF THE HESSENBERG MATRIX H_{m} USING FOR EXAMPLE QR FACTORIZATION
 
 This provides estimates for m << n Ritz vectors (eigenvector approx.) and Ritz values (eigenvalues approx.) The Ritz values typically converge to the "extreme eigenvalues of the spectrum". So the extreme ritz values (smallest and largest) of the Arnoldi iteration become better and better approximations of the true eigenvaues of A as the dimensionality of our Krylov subspace grows
 */
//Forward Declarations
double dot_product(const double* x, const double* y, int n);
void mat_vec_mult(double** A, const double* x, double* result, int n);
typedef struct {
    double** H;
    double** Q;
} Arnoldi; //want to return two matrices so create a struct to package both these
Arnoldi* arnoldi(double** A, double* u, int m, int ulen);
//=====================MAIN FUNCTION================================================
int main(void){
    int n = 10;//nrows of A and b
    int m = 10; //number of Arnoldi iterations
    double** A = malloc(n*sizeof(double*));
    for(int i = 0; i<n; i++){
        A[i] = malloc(n*sizeof(double));
    }
    //initialise example matrix A
    A[0][0]= 3; A[0][1]= 8; A[0][2]= 7; A[0][3]= 3; A[0][4]= 3; A[0][5]= 7; A[0][6]= 2; A[0][7]= 3; A[0][8]= 4; A[0][9]= 8;
    A[1][0]= 5; A[1][1]= 4; A[1][2]= 1; A[1][3]= 6; A[1][4]= 9; A[1][5]= 8; A[1][6]= 3; A[1][7]= 7; A[1][8]= 1; A[1][9]= 9;
    A[2][0]= 3; A[2][1]= 6; A[2][2]= 9; A[2][3]= 4; A[2][4]= 8; A[2][5]= 6; A[2][6]= 5; A[2][7]= 6; A[2][8]= 6; A[2][9]= 6;
    A[3][0]= 5; A[3][1]= 3; A[3][2]= 4; A[3][3]= 7; A[3][4]= 4; A[3][5]= 9; A[3][6]= 2; A[3][7]= 3; A[3][8]= 5; A[3][9]= 1;
    A[4][0]= 4; A[4][1]= 4; A[4][2]= 2; A[4][3]= 1; A[4][4]= 7; A[4][5]= 4; A[4][6]= 2; A[4][7]= 2; A[4][8]= 4; A[4][9]= 5;
    A[5][0]= 4; A[5][1]= 2; A[5][2]= 8; A[5][3]= 6; A[5][4]= 6; A[5][5]= 5; A[5][6]= 2; A[5][7]= 1; A[5][8]= 1; A[5][9]= 2;
    A[6][0]= 2; A[6][1]= 8; A[6][2]= 9; A[6][3]= 5; A[6][4]= 2; A[6][5]= 9; A[6][6]= 4; A[6][7]= 7; A[6][8]= 3; A[6][9]= 3;
    A[7][0]= 9; A[7][1]= 3; A[7][2]= 2; A[7][3]= 2; A[7][4]= 7; A[7][5]= 3; A[7][6]= 4; A[7][7]= 8; A[7][8]= 7; A[7][9]= 7;
    A[8][0]= 9; A[8][1]= 1; A[8][2]= 9; A[8][3]= 3; A[8][4]= 3; A[8][5]= 1; A[8][6]= 2; A[8][7]= 7; A[8][8]= 7; A[8][9]= 1;
    A[9][0]= 9; A[9][1]= 3; A[9][2]= 2; A[9][3]= 2; A[9][4]= 6; A[9][5]= 4; A[9][6]= 4; A[9][7]= 7; A[9][8]= 3; A[9][9]= 5;
    
    //initialise example vector b
    double* b = malloc(n*sizeof(double));
    b[0] = 0.757516242460009; b[1] = 2.734057963614329; b[2] = -0.555605907443403; b[3] = 1.144284746786790; b[4] = 0.645280108318073; b[5] = -0.085488474462339; b[6] = -0.623679022063185; b[7] = -0.465240896342741; b[8] = 2.382909057772335; b[9] = -0.120465395885881;
    
    Arnoldi* result = arnoldi(A, b, m, n);
    printf("Basis Vector number %d \n", m);
    for(int i = 0; i<n; i++){
        printf(" %f\n", result->Q[m-1][i]);
    }
    //free the memory
    for (int i = 0; i < n; i++) {
            free(A[i]);
        }
        free(A);

        for (int i = 0; i < m + 1; i++) {
            free(result->Q[i]);
            free(result->H[i]);
        }
        free(result->Q);
        free(result->H);
        free(result);

        return 0;
}

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

    

