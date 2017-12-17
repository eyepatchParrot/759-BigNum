

#include <vector>
#include <set>


typedef unsigned long long ulong;
typedef unsigned int size_t;




	ulong 1stNumber, 2ndNumber, result;           

	std::vector<ulong> 1stNumberSplit;
	std::vector<ulong> 2ndNumberSplit;

	std::vector<ulong> 1stPolynomial;
	std::vector<ulong> 2ndPolynomial;

	std::vector<ulong> calcPolynomial; 

	std::set<int> point;

	size_t 1stLength;
	size_t 2ndLength;

	size_t 1stbaseLength;
	size_t 2ndbaseLength;




	void _Toom_Cook_3();
	void _Toom_Cook_3(const ulong&, const ulong&);

	
	__global__ void Multiply();
	
	
	__global__ void Split();
	__global__ void Evaluation();
	__global__ void PointWiseMult();
	__global__ void Interpolation();
	__global__ void Recomposition(); 

	
	void computeLength(ulong);
	ulong separate(std::vector<ulong>&, ulong, const size_t&);
	void merge(std::vector<ulong>&);
	
	void Evaluating(const std::vector<ulong>&, std::vector<ulong>&); 

	ulong Karatsuba(ulong, ulong); 
	ulong leftSplit(ulong number, const int& length);
	ulong rightSplit(ulong number, const int& length);

	int ComputePolynomialLength(ulong);
	int numberPower(const ulong&, const ulong&, const ulong&) const;



