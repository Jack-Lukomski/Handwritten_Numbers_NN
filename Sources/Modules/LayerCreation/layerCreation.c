#include "layerCreation.h"

void vCreateNNLayerMatricies (void)
{
    FILE * fp = fopen("InputLayer/InputWeights.csv", "r");

    if (!fp) {
        printf("ERROR\n");
    }

    char ch;

    while ((ch = fgetc(fp)) != EOF)
    {
        printf("%c", ch);
    }
}