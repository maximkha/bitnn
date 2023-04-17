#include <cstdlib> 
#include <cstdint>
#include <iostream>

int main(int argc, char const *argv[])
{
    size_t is_true = 0;
    size_t total = 10000;
    for (size_t i = 0; i < total; i++)
    {
        if ((rand() % 1) == 0) is_true++;
    }
    
    std::cout << is_true << "/" << total << "=" << ((float)is_true/ (float)total) << "\n";

    return 0;
}