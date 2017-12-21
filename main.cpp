
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <limits>


#include "_Toom_Cook__3.h"

using namespace std;


int main()
{
	_Toom_Cook_3 first(123456 , 654321);
	first.Multiply();


	return 0;
}
