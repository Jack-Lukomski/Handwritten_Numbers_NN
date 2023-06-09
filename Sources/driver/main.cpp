#include <iostream>
#include "../include/Layer.hpp"

int main () 
{
    // Create a matrix
    arma::mat A = {{1.0, 2.0, 3.0},
                   {4.0, 5.0, 6.0},
                   {7.0, 8.0, 9.0}};

    arma::mat B(arma::size(3, 1));
    B << 1.0 << 4.5 << 8.8;

    Layer hl1(A, B, LayerType::HiddenLayer);
    hl1.printLayer();

    return 0;    
}