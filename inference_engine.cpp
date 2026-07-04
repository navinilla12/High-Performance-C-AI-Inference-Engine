#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

inline float relu(float x) {
    return x > 0 ? x : 0;
}

void matmul(const vector<vector<float>>& A,
            const vector<vector<float>>& B,
            vector<vector<float>>& C) {

    int n = A.size();
    int p = B.size();
    int m = B[0].size();

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < p; k++) {
            float a = A[i][k];
            for (int j = 0; j < m; j++) {
                C[i][j] += a * B[k][j];
            }
        }
    }
}

void forward(vector<vector<float>>& X,
             vector<vector<float>>& W,
             vector<vector<float>>& out) {

    matmul(X, W, out);

    for (auto &row : out)
        for (auto &v : row)
            v = relu(v);
}

int main() {

    vector<vector<float>> X(128, vector<float>(64, 1.0));
    vector<vector<float>> W(64, vector<float>(32, 0.5));
    vector<vector<float>> out(128, vector<float>(32, 0));

    auto start = chrono::high_resolution_clock::now();

    forward(X, W, out);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> diff = end - start;

    cout << "Inference Time: " << diff.count() << " sec" << endl;

    cout << "Output sample: " << out[0][0] << endl;
}
