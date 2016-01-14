#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

class BLSVM
{
public:

	typedef unsigned char byte;
	
	BLSVM()
	{
		dim = 0;
		weight = NULL;
	}
	~BLSVM()
	{
		if (weight != NULL)
			delete[] weight;
	}

	void BLSVMTrain(int m, int n, float *X, int *y, float c, int ker = 0, float arg = 0.0f);
	bool saveModel(const std::string model);
	bool readModel(const std::string model);
	bool predict(int n, float *X, int &y);
	bool predict(int m, int n, float *X, int *y);

	float *weight;
	int dim;

private:
	 void KernelMatrix(int m, int n, float *X, int *y, int ker, float arg, float *K);
	float KernelFunction(int n, float *x1, float *x2, int ker, float arg);
	  int FSMO(int n, float *A, int *b, float *x, float xh, float tol);
	 
};

bool BLSVM::saveModel(const std::string model)
{
	std::fstream file;
	file.open(model, std::fstream::binary | std::fstream::out);
	if (!file.is_open())
	{
		std::cerr << "fail to create " << model << std::endl;
		return false;
	}

	if (dim <= 0 || weight == NULL)
	{
		std::cerr << "the model has not been trained yet" << std::endl;
		return false;
	}

	file << dim;
	file.write((const char*)weight, sizeof(float)*(dim+1));

	file.close();
	return true;
}

bool BLSVM::readModel(const std::string model)
{
	std::fstream file;
	file.open(model, std::fstream::binary | std::fstream::in);
	if (!file.is_open())
	{
		std::cerr << "fail to open " << model << std::endl;
		return false;
	}

	file >> dim;
	if (dim <= 0)
	{
		std::cerr << "invalid model file" << std::endl;
		return false;
	}

	if (weight != NULL)
	{
		delete[] weight;
		weight = NULL;
	}
	weight = new float[dim + 1];
	std::streamsize size = file.readsome((char*)weight, sizeof(float)* (dim+1));
	
	switch (size)
	{
	case std::fstream::eofbit:
		std::cerr << "the value of dimensionality is larger than the number of weight stored" << std::endl;
		std::cerr << "dim = " << size << ", number of weights = " << size << std::endl;
		return false;
	case std::fstream::failbit:
		std::cerr << "the value of dimensionality is less than the number of weight stored" << std::endl;
		std::cerr << "dim = " << size << std::endl;
		return false;
	case std::fstream::badbit:
		std::cerr << "the model file may be broken" << std::endl;
		return false;
	default:
		break;
	}

	file.close();
	return true;
}

bool BLSVM::predict(int n, float *X, int &y)
{
	if (dim == 0 || weight == NULL)
	{
		std::cerr << "load a model or train a model first" << std::endl;
		return false;
	}

	if (dim != n)
	{
		std::cerr << "dimensionality does not match" << std::endl;
		std::cerr << "input dim = " << n << ", model dim = " << dim << std::endl;
		return false;
	}

	if (X == NULL)
	{
		std::cerr << "invalid test data" << std::endl;
		return false;
	}

	int i = 0;
	float pre = 0.0f;
	try
	{
		for (; i < n; i++)
			pre += X[i] * weight[i];
		pre += weight[n];
	}
	catch (const std::exception& e)
	{
		std::cerr << "a standard exception occurs at index "<< i << " :" << std::endl << e.what() << std::endl;
		return false;
	}

	y = ((pre >= 0) ? 1 : -1);

	return true;
}

bool BLSVM::predict(int m, int n, float *X, int *y)
{
	if (dim == 0 || weight == NULL)
	{
		std::cerr << "load a model or train a model first" << std::endl;
		return false;
	}

	if (dim != n)
	{
		std::cerr << "dimensionality does not match" << std::endl;
		std::cerr << "input dim = " << m << ", model dim = " << dim << std::endl;
		return false;
	}

	if (X == NULL)
	{
		std::cerr << "invalid test data" << std::endl;
		return false;
	}

	if (m < 0)
	{
		std::cerr << "invalid number of test data" << std::endl;
		return false;
	}

	float *pX = X;
	int i, j;
	i = j = 0;
	try
	{
		for (; i < m; i++, pX += n)
		{
			float pre = 0.0f;
			for (j = 0; j < n; j++)
				pre += weight[j] * pX[j];
			pre += weight[n];
			y[i] = (pre >= 0 ? 1 : -1);
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "a standard exception occurs at" << i << "-th data" << ", index " << j << " :" << std::endl << e.what() << std::endl;
		return false;
	}

	return true;
}

void BLSVM::BLSVMTrain(int m, int n, float *X, int *y, float c, int ker, float arg)
{
	if (ker < 0 || ker > 2)
	{
		std::cerr << "wrong kernel index\n0: linear, 1: polynomial, 2:gaussian" << std::endl;
		return;
	}

	dim = n;
	if (weight != NULL)
	{
		delete[] weight;
		weight = NULL;
	}
	weight = new float[dim + 1];
	memset(weight, 0, sizeof(float)*(dim + 1));

	float *A = new float[m*m];
	float *a = new float[m];
	byte  *s = new byte[m];

	KernelMatrix(m, n, X, y, ker, arg, A);

	const float tol = 0.001f;
	int t = FSMO(m, A, y, a, c, tol);

	int nsv = 0, i;
	for (i = 0; i<m; i++)
	{
		if (a[i]>tol)
		{
			nsv++;
			if (a[i] < c - tol) s[i] = 2;
			else s[i] = 1;
		}
		else s[i] = 0;
	}
	printf("ker = %d, arg = %.2f, fsmo = %d, nsv = %d\n", ker, arg, t, nsv);

	// linear svm
	float *w = new float[n];
	for (i = 0; i < n; i++)
	{
		float temp = 0;
		for (int j = 0; j<m; j++) if (s[j]>0) temp += y[j] * a[j] * X[n*j + i];
		w[i] = temp;
	}

	float b = 0, nb = 0;
	for (i = 0; i < m; i++)
	{
		if (s[i] != 2) continue;
		float temp = 0;
		for (int j = 0; j<n; j++) temp += w[j] * X[n*i + j];
		b += y[i] - temp;
		nb++;
	}
	if (nb>0) b /= nb;

	memcpy(weight, w, n*sizeof(float));
	weight[n] = b;
	delete[]w;

	delete[]s;
	delete[]a;
	delete[]A;
}

//-------------------------------------------------------------------------------------------------
// Kernel matrix
// Km*m, Xm*n
// y==0, kij = k(xi, xj)
// y!=0, kij = yi*yj*k(xi, xj)
void BLSVM::KernelMatrix(int m, int n, float *X, int *y, int ker, float arg, float *K)
{
	for (int i = 0; i < m; i++)
	{
		float *xi = X + n*i;
		for (int j = 0; j < i; j++)
		{
			float *xj = X + n*j;
			if (y == 0) K[m*i + j] = K[m*j + i] = KernelFunction(n, xi, xj, ker, arg);
			else K[m*i + j] = K[m*j + i] = (y[i] * y[j])*KernelFunction(n, xi, xj, ker, arg); //+/-1
		}
		K[m*i + i] = KernelFunction(n, xi, xi, ker, arg);
	}
}

//-------------------------------------------------------------------------------------------------
// Kernel function
// 0 linear: x'*x2
// 1 polynomial: (1+x1'*x2)^p
// 2 Gaussian: exp(-||x1-x2||^2/(2*r^2))
float BLSVM::KernelFunction(int n, float *x1, float *x2, int ker, float arg)
{
	float f = 0;

	switch (ker)
	{
		int i;
		float x12, dx;
	case 0:
		for (i = 0; i < n; i++) f += x1[i] * x2[i];
		break;

	case 1:
		x12 = 1;
		for (i = 0; i < n; i++) x12 += x1[i] * x2[i];
		f = (float)std::pow(x12, int(arg + 0.5));
		break;

	case 2:
		dx = 0;
		for (i = 0; i < n; i++) dx += (x1[i] - x2[i])*(x1[i] - x2[i]);
		f = (float)std::exp(-dx / (2 * arg*arg));
		break;
	}

	return f;
}


//-------------------------------------------------------------------------------------------------
// Fast Sequential Minimal Optimization (FSMO)
//    min  Q(x) = 0.5*x'*A*x - 1'*x, 
//    s.t. b'*x = 0, 0 <= x <= xh.
int BLSVM::FSMO(int n, float *A, int *b, float *x, float xh, float tol)
{
	float *dQ = new float[n];
	bool *b1 = new bool[n];
	bool *b2 = new bool[n];

	//initialize dQ, b1 and b2
	const float ZERO = 0.000001f;
	const float XMAX = xh - ZERO;
	memset(x, 0, n*sizeof(float));
	for (int i = 0; i < n; i++)
	{
		dQ[i] = -1;
		for (int j = 0; j<n; j++) dQ[i] += A[i*n + j] * x[j];
		if ((b[i]<0 && x[i]<XMAX) || (b[i]>0 && x[i]>ZERO)) b1[i] = true;
		else b1[i] = false;
		if ((b[i]<0 && x[i]>ZERO) || (b[i]>0 && x[i] < XMAX)) b2[i] = true;
		else b2[i] = false;
	}

	//begin iterations of x and dQ
	int t;
	for (t = 1; t < 10000; t++)
	{
		//find two variables that produce a maximal reduction in Q
		float dQ1 = -1e10, dQ2 = 1e10;
		int i, i1, i2;
		for (i = 0; i < n; i++)
		{
			float bdQ = b[i] * dQ[i];
			if (b1[i] && dQ1<bdQ){ dQ1 = bdQ; i1 = i; }
			if (b2[i] && dQ2>bdQ){ dQ2 = bdQ; i2 = i; }
		}
		if (dQ1 - dQ2 < tol) break;

		//optimize x(i1) and x(i2)
		float x1, x2, s;
		s = (float)(b[i1] * b[i2]);
		x2 = x[i2] + b[i2] * (dQ1 - dQ2) / (A[i1*n + i1] - 2 * s*A[i1*n + i2] + A[i2*n + i2]);

		//check its bounds
		float L, H;
		if (s < 0)
		{
			L = std::max(0.0f, x[i2] - x[i1]);
			H = std::min(xh, xh + x[i2] - x[i1]);
		}
		else
		{
			L = std::max(0.0f, x[i2] + x[i1] - xh);
			H = std::min(xh, x[i2] + x[i1]);
		}
		if (x2<L) x2 = L;
		if (x2>H) x2 = H;
		if (s < 0) x1 = x[i1] - x[i2] + x2;
		else x1 = x[i1] + x[i2] - x2;

		//update gradient dQ and solution x
		float dx1 = x1 - x[i1], dx2 = x2 - x[i2];
		for (i = 0; i<n; i++) dQ[i] += dx1*A[i*n + i1] + dx2*A[i*n + i2];
		x[i1] = x1;
		x[i2] = x2;

		//update b1 and b2
		if ((b[i1]<0 && x1<XMAX) || (b[i1]>0 && x1>ZERO)) b1[i1] = true;
		else b1[i1] = false;
		if ((b[i1]<0 && x1>ZERO) || (b[i1]>0 && x1<XMAX)) b2[i1] = true;
		else b2[i1] = false;
		if ((b[i2]<0 && x2<XMAX) || (b[i2]>0 && x2>ZERO)) b1[i2] = true;
		else b1[i2] = false;
		if ((b[i2]<0 && x2>ZERO) || (b[i2]>0 && x2 < XMAX)) b2[i2] = true;
		else b2[i2] = false;
	}

	delete[]b1;
	delete[]b2;
	delete[]dQ;

	return t;
}