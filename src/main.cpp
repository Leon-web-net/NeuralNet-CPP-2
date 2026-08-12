#include <iostream>
#include <fstream>
#include <string>



int main()
{
	std::cout << "Data dir: " << DATA_DIR << std::endl;
	std::ifstream f(std::string(DATA_DIR) + "/train.csv");
	std::cout << (f.is_open() ? "found train.csv\n" : "train.csv not found\n");


	return 0;
}
