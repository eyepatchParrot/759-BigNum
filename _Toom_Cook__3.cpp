#include "_Toom_Cook__3.h"

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#inlcude <omp.h>

// Constructor
_Toom_Cook_3::_Toom_Cook_3() 
: 1stNumber(0), 2ndNumber(0), result(0), 	
		1stNumberSplit(std::vector<ulong>()), 
		2ndNumberSplit(std::vector<ulong>()), 
				
		1stPolynomial(std::vector<ulong>(5)),
		2ndPolynomial(std::vector<ulong>(5)),
		calcPolynomial(std::vector<ulong>(5)),

		point(std::set<int>()), 
				
		1stLength(0), 2ndLength(0), 
		1stbaseLength(0), 2ndbaseLength(0)
{
	point.insert(0);
	point.insert(1);
	point.insert(-1);
	point.insert(-2);
	// infinity
	point.insert(-1);
}
// Constructor
_Toom_Cook_3::_Toom_Cook_3(const ulong& firstUserNumber, const ulong& secondUserNumber)
: 1stNumber(firstUserNumber), 2ndNumber(secondUserNumber), result(0), 	
		1stNumberSplit(std::vector<ulong>()), 
		2ndNumberSplit(std::vector<ulong>()), 
				
		1stPolynomial(std::vector<ulong>(5)),
		2ndPolynomial(std::vector<ulong>(5)),
		calcPolynomial(std::vector<ulong>(5)),

		point(std::set<int>()), 
				
		1stLength(0), 2ndLength(0), 
		1stbaseLength(0), 2ndbaseLength(0)
{
	point.insert(0);
	point.insert(1);
	point.insert(-1);
	point.insert(-2);
	// infinity
	point.insert(-1);

}
// Multiply
void _Toom_Cook_3::Multiply() 
{
	Split();
	Evaluation();
	PointWiseMult();
	Interpolation();
	Recomposition();
}
// Split
void _Toom_Cook_3::Split()
{
	const int _Toom_Cook_33 = 3;
	
	computeLength(this->1stNumber);
	1stbaseLength = (this->1stLength + 2) / _Toom_Cook_33; 
	ulong tempNumber(0);
	
	tempNumber = separate(1stNumberSplit, 1stNumber, 1stbaseLength);
	tempNumber = separate(1stNumberSplit, tempNumber, 1stbaseLength);
	tempNumber = separate(1stNumberSplit, tempNumber, 1stbaseLength);

	computeLength(this->2ndNumber);
	2ndbaseLength = (this->2ndLength + 2) / _Toom_Cook_33;
	tempNumber = separate(2ndNumberSplit, 2ndNumber, 2ndbaseLength);
	tempNumber = separate(2ndNumberSplit, tempNumber, 2ndbaseLength);
	tempNumber = separate(2ndNumberSplit, tempNumber, 2ndbaseLength);

}
// Evaluation
void _Toom_Cook_3::Evaluation()
{
	Evaluating(1stNumberSplit, 1stPolynomial);
	Evaluating(2ndNumberSplit, 2ndPolynomial);
}
// PointWiseMult
void _Toom_Cook_3::PointWiseMult()
{
	// fake variable to turn off error
	ulong first(12345), second(12345);

	Karatsuba(first, second);
}
// Interpolation
void _Toom_Cook_3::Interpolation()
{
	calcPolynomial[3] = (calcPolynomial[3] + calcPolynomial[1]) / 3;
	calcPolynomial[1] = (calcPolynomial[1] - calcPolynomial[2]) / 2;
	calcPolynomial[2] = calcPolynomial[2] - calcPolynomial[0];
	calcPolynomial[3] = ((calcPolynomial[2] - calcPolynomial[3])/2) + (2 * calcPolynomial[4]);
	
	calcPolynomial[2] = calcPolynomial[2] + calcPolynomial[1] - calcPolynomial[4];
	calcPolynomial[1] = calcPolynomial[1] - calcPolynomial[3];

}
// Recomposition
void _Toom_Cook_3::Recomposition()
{
}
// computeLength
void _Toom_Cook_3::computeLength(ulong number)
{
	if (1stLength == 0)
	{
		while (number / 10 )
		{
			number /= 10;
			++1stLength;
		}
	}
	else
	{
		while (number / 10 )
		{
			number /= 10;
			++2ndLength;
		}
	}
}
// separate
ulong _Toom_Cook_3::separate(std::vector<ulong>& numberSplit, 
											ulong number, const size_t& baseLength)
{
	size_t countLoop(0);
	ulong tempNumber(number);
	std::vector<ulong> tempSplit;

	while (countLoop < baseLength)
	{
		tempSplit.push_back(tempNumber % 10); 
		tempNumber /= 10;
		++countLoop;
	}
	
	std::reverse(tempSplit.begin(), tempSplit.end());
	merge(tempSplit);

	numberSplit.push_back(tempSplit[0]);
	
	return tempNumber;
}
// merge
void _Toom_Cook_3::merge(std::vector<ulong>& number)
{
	size_t loop(0);
	 
	while (loop < number.size() - 1)
	{
		number[0] = (number[0] * 10) + number[loop + 1]; 
		++loop;
	}

	number.erase(number.begin() + 1, number.end());
	std::vector<ulong>(number).swap(number);
}
// Evaluating
void _Toom_Cook_3::Evaluating(const std::vector<ulong>& number, 
					std::vector<ulong>& polynomial)
{
	// Bodrato approach
	if (number.size() > 2 && polynomial.size() > 4)
	{
		polynomial[0] = number[0];
		polynomial[1] = polynomial[0] + number[1];
		polynomial[2] = polynomial[0] - number[1];
		polynomial[3] = ((polynomial[2] + number[2]) * 2) - number[0];
		polynomial[4] = number[2];
	}

/*
General Approach
p(0) = m0 + m1(0) + m2(0 ^ 2)
     = m0
p(1) = m0(1) + m1(1) + m2(1 ^ 2)
p(-1) = m0(-1) + m1(-1) + m2(-1 ^ 2)
      = m0 - m1 + m2
p(-2) = m0(-2) + m1(-2) + m2(-2 ^ 2)
      = m0 -2m1 + 4m2
P(infinity) = m2
	*/
}
//Karatsuba
ulong _Toom_Cook_3::Karatsuba(ulong first, ulong second)
{
	const int MINLENGTH = 2;

	if (this->1stbaseLength > MINLENGTH && 
		this->2ndbaseLength > MINLENGTH)
	{
		if (ComputePolynomialLength(first) > MINLENGTH && ComputePolynomialLength(second) > MINLENGTH)
		{
			int loop(0);
			while (loop < static_cast<int>(1stPolynomial.size()) &&
				static_cast<int>(2ndPolynomial.size()) )
			{
				ulong x1 = leftSplit(1stPolynomial[loop], ComputePolynomialLength(1stPolynomial[loop]));
				ulong y1 = leftSplit(2ndPolynomial[loop], ComputePolynomialLength(2ndPolynomial[loop]));
				
				// wrong here
				ulong x0 = rightSplit(1stPolynomial[loop], ComputePolynomialLength(1stPolynomial[loop]) - ComputePolynomialLength(x1));
				ulong y0 = rightSplit(2ndPolynomial[loop], ComputePolynomialLength(2ndPolynomial[loop]) - ComputePolynomialLength(y1));
				
				int lengthPower = numberPower(1stPolynomial[loop], x1, x0);

				ulong X = Karatsuba(x1, y1);
				ulong Y = Karatsuba(x0, y0);
				ulong Z = Karatsuba(x1 + x0, y1 + y0);
				Z = Z - X - Y;
			
				calcPolynomial[loop] = (X * static_cast<unsigned long>(pow(10.0, 2.0 * lengthPower))) +  
					(Z * static_cast<unsigned long>(pow(10.0, lengthPower))) + Y;

				++loop;
			}
		}
		else
		{
			return first * second;
		}
		
	}
	else // Base Length < 4
	{
		if (1stPolynomial.size() > 4 &&
			2ndPolynomial.size() > 4 && 
			calcPolynomial.size() > 4)
		{
			calcPolynomial[0] = 1stPolynomial[0] * 2ndPolynomial[0];
			calcPolynomial[1] = 1stPolynomial[1] * 2ndPolynomial[1];
			calcPolynomial[2] = 1stPolynomial[2] * 2ndPolynomial[2];
			calcPolynomial[3] = 1stPolynomial[3] * 2ndPolynomial[3];
			calcPolynomial[4] = 1stPolynomial[4] * 2ndPolynomial[4];
		}
	}

	

	return 1;
}
// ComputePolynomialLength
int _Toom_Cook_3::ComputePolynomialLength(ulong number)
{
		int len(0);

	// Check length of integer algorithm
	// * 10 = To shift left 1 digit in number
	// % 10  = To get last digit of number
	while (number >= 1)
	{
		number /= 10;
		++len;
	}

	return len;
}
// leftSplit
ulong _Toom_Cook_3::leftSplit(ulong number, const int& length)
{
	int middle = length / 2;
	std::vector<ulong> remainder(0);

	// To get most significant digit
	while (number >= 10)
	{
		remainder.push_back(number % 10);
		number /= 10;
	}
	
	std::reverse(remainder.begin(), remainder.end());

	ulong result(number);int remLoop(0);
	
	#pragma omp parallel for  default(shared) private(loop) schedule(static,chunk) reduction(+:result) 
   
	for (int loop = 0;loop < middle - 1;++loop)
	{
		if (remLoop < static_cast<int>(remainder.size()))
		{
			result = result * 10 + remainder[remLoop];
		}
		++remLoop;
	}

	return result;
}
// rightSplit
ulong _Toom_Cook_3::rightSplit(ulong number, const int& length)
{
	ulong remainder(0), multiply(1);
	ulong result(0);
	#pragma omp parallel for  default(shared) private(loop) schedule(static,chunk) reduction(+:result) 
	for (int loop = 0; loop < length;++loop)
	{
		remainder = number % 10;
		number /= 10;
		result += remainder * multiply ;
		multiply *= 10;
	}

	return result;
}
// numberPower
int _Toom_Cook_3::numberPower(const ulong& first, const ulong& x1, 
								const ulong& y1) const
{
	int lengthPower(1);

	const int base(10);
	
	while (first - y1 != (x1 * (pow(static_cast<double>(base), 
		static_cast<int>(lengthPower)))) )
	{
		++lengthPower;
	}

	return lengthPower;
}
