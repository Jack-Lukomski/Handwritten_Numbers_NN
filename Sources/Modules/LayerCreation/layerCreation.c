#include "layerCreation.h"

void vCreateNNLayerMatricies (void)
{
    LayerCSVFiles * allFiles = xConstructLayerCSVFiles();

    fileStructure InputWeightsMatrixSize = xGetFileStructure(allFiles->InputWeights);
    double * InputWeightsMatrixData = xGetFileData(allFiles->InputWeights, InputWeightsMatrixSize);
    //INPUT_LAYER_WEIGHTS = (Matrix *) malloc(sizeof(Matrix));
    INPUT_LAYER_WEIGHTS = xCreateMatrix(InputWeightsMatrixSize.rows, InputWeightsMatrixSize.cols, InputWeightsMatrixData);
    //vPrintMatrix(INPUT_LAYER_WEIGHTS);
}

static LayerCSVFiles * xConstructLayerCSVFiles (void)
{
    LayerCSVFiles * newCSVObj = (LayerCSVFiles *) malloc(sizeof(LayerCSVFiles));
    newCSVObj->InputWeights = fopen("../../../Sources/Modules/LayerCreation/InputLayer/InputWeights.csv", "r");
    newCSVObj->HiddenLayerWeights = (FILE **) malloc(NUM_HIDDEN_LAYERS * sizeof(FILE *));
    char filename[100];

    for (uint16_t currHLFile = 0; currHLFile < NUM_HIDDEN_LAYERS; currHLFile++)
    {
        sprintf(filename, "../../../Sources/Modules/LayerCreation/HiddenLayers/HiddenWeights%d.csv", currHLFile);
        newCSVObj->HiddenLayerWeights[currHLFile] = fopen(filename, "r");
    }

    newCSVObj->HiddenBiases = fopen("../../../Sources/Modules/LayerCreation/HiddenLayers/HiddenBiases.csv", "r");
    newCSVObj->OutputWeights = fopen("../../../Sources/Modules/LayerCreation/OutputLayer/OutputWeights.csv", "r");
    newCSVObj->OutputBiases = fopen("../../../Sources/Modules/LayerCreation/OutputLayer/OutputBiases.csv", "r");

    return newCSVObj;
}

static double * xGetFileData (FILE * f, fileStructure currFileStructure)
{
    uint16_t currBuf = 0;
    double currBufVal;
    uint16_t matrixSize = currFileStructure.cols*currFileStructure.rows;
    double * currFileData = (double *) malloc(matrixSize * sizeof(double));

    while (fscanf(f, "%lf,", &currBufVal) == 1 && currBuf < matrixSize)
    {
        currFileData[currBuf] = currBufVal;
        currBuf++;
    }

    return currFileData;
}

static fileStructure xGetFileStructure(FILE *f) 
{
    fileStructure fsRetVal;
    uint16_t rows = 0;
    uint16_t cols = 0;
    bool bColDone = false;
    char currBuf;

    while((currBuf = fgetc(f)) != EOF)
    {
        if (currBuf == ',' && !bColDone)
        {
            cols++;
        }
        if (currBuf == '\n')
        {
            rows++;
            bColDone = true;
        }
    }

    fsRetVal.cols = cols+1;
    fsRetVal.rows = rows+1;

    fseek(f, 0, SEEK_SET);

    return fsRetVal;
}
