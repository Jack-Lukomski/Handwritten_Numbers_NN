#include "matrix.h"


int main(void)
{
    double m1[9] = {1.2, 34.2, 
                    21.3, 3.4};
                      
    double m2[9] = {1.6, 3.2, 
                    21.3, 4.5};

    double m3[9] = {1.4, 1.1, 5.4, 
                    4.5, 6.5, 6.7,
                    7.7, 9.0, 34.2};

    double m4[3] = {1.2, 4.4, 44.3};

    Matrix * ma1 = xCreateMatrix(2, 2, m1);
    Matrix * ma2 = xCreateMatrix(2, 2, m2);
    Matrix * ma3 = xCreateMatrix(3, 3, m3);
    Matrix * ma4 = xCreateMatrix(1, 3, m4);

    Matrix * dotp = xDotProduct(ma4, ma3);

    vPrintMatrix(dotp);

    return 0;
}