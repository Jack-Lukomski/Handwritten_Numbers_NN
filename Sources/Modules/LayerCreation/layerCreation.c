#include "layerCreation.h"

void vCreateNNLayerMatricies (void)
{
    LayerCSVFiles * allFiles = xConstructLayerCSVFiles();

    fileStructure InputWeightsMatrixSize = xGetFileStructure(allFiles->InputWeights);
    double * InputWeightsMatrixData = xGetFileData(allFiles->InputWeights, InputWeightsMatrixSize);
    INPUT_LAYER_WEIGHTS = xCreateMatrix(InputWeightsMatrixSize.rows, InputWeightsMatrixSize.cols, InputWeightsMatrixData);
    
    HIDDEN_LAYER_WEIGHTS[NUM_HIDDEN_LAYERS];
    for (uint16_t currHL = 0; currHL < NUM_HIDDEN_LAYERS; currHL++)
    {
        fileStructure currHiddenWeightsMatrixSize = xGetFileStructure(allFiles->HiddenLayerWeights[currHL]);
        double * currHiddenWeightsMatrixData = xGetFileData(allFiles->HiddenLayerWeights[currHL], currHiddenWeightsMatrixSize);
        HIDDEN_LAYER_WEIGHTS[currHL] = xCreateMatrix(currHiddenWeightsMatrixSize.rows, currHiddenWeightsMatrixSize.cols, currHiddenWeightsMatrixData);
    }
    
    HIDDEN_LAYER_BIASES[NUM_HIDDEN_LAYERS];
    for (uint16_t currHL = 0; currHL < NUM_HIDDEN_LAYERS; currHL++)
    {
        fileStructure currHiddenBiasMatrixSize = xGetFileStructure(allFiles->HiddenBiases[currHL]);
        double * currHiddenBiasMatrixData = xGetFileData(allFiles->HiddenBiases[currHL], currHiddenBiasMatrixSize);
        HIDDEN_LAYER_BIASES[currHL] = xCreateMatrix(currHiddenBiasMatrixSize.rows, currHiddenBiasMatrixSize.cols, currHiddenBiasMatrixData);
    }

    fileStructure OutputWeightsMatrixSize = xGetFileStructure(allFiles->OutputWeights);
    double * OutputWeightsMatrixData = xGetFileData(allFiles->OutputWeights, OutputWeightsMatrixSize);
    OUTPUT_LAYER_WEIGHTS = xCreateMatrix(OutputWeightsMatrixSize.rows, OutputWeightsMatrixSize.cols, OutputWeightsMatrixData);

    fileStructure OutputBiasesMatrixSize = xGetFileStructure(allFiles->OutputBiases);
    double * OutputBiasesMatrixData = xGetFileData(allFiles->OutputBiases, OutputBiasesMatrixSize);
    OUTPUT_LAYER_BIASES = xCreateMatrix(OutputBiasesMatrixSize.rows, OutputBiasesMatrixSize.cols, OutputBiasesMatrixData);
}

static LayerCSVFiles * xConstructLayerCSVFiles (void)
{
    LayerCSVFiles * newCSVObj = (LayerCSVFiles *) malloc(sizeof(LayerCSVFiles));
    newCSVObj->InputWeights = fopen("../../../Libaries/NeuralC/Sources/Modules/LayerCreation/InputLayer/InputWeights.csv", "r");
    newCSVObj->HiddenLayerWeights = (FILE **) malloc(NUM_HIDDEN_LAYERS * sizeof(FILE *));
    newCSVObj->HiddenBiases = (FILE **) malloc(NUM_HIDDEN_LAYERS * sizeof(FILE *));
    char filename[100];

    for (uint16_t currHLFile = 0; currHLFile < NUM_HIDDEN_LAYERS; currHLFile++)
    {
        sprintf(filename, "../../../Libaries/NeuralC/Sources/Modules/LayerCreation/HiddenLayers/HiddenWeights%d.csv", currHLFile);
        newCSVObj->HiddenLayerWeights[currHLFile] = fopen(filename, "r");
    }

    for (uint16_t currHLFile = 0; currHLFile < NUM_HIDDEN_LAYERS; currHLFile++)
    {
        sprintf(filename, "../../../Libaries/NeuralC/Sources/Modules/LayerCreation/HiddenLayers/HiddenBiases%d.csv", currHLFile);
        newCSVObj->HiddenBiases[currHLFile] = fopen(filename, "r");
    }

    //newCSVObj->HiddenBiases = fopen("../../../Sources/Modules/LayerCreation/HiddenLayers/HiddenBiases.csv", "r");

    newCSVObj->OutputWeights = fopen("../../../Libaries/NeuralC/Sources/Modules/LayerCreation/OutputLayer/OutputWeights.csv", "r");
    newCSVObj->OutputBiases = fopen("../../../Libaries/NeuralC/Sources/Modules/LayerCreation/OutputLayer/OutputBiases.csv", "r");

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

void vUpdateInputLayerWeights (Matrix * inputLayerWeights, NerualNetwork * NN)
{
    FILE * ilWeights_ptr = fopen("../../../Libaries/NeuralC/Sources/Modules/LayerCreation/InputLayer/InputWeights.csv", "w");

    for (uint16_t currWeight = 0; currWeight < inputLayerWeights->cols*inputLayerWeights->rows; currWeight++)
    {
        if (currWeight == inputLayerWeights->cols*inputLayerWeights->rows-1)
        {
            fprintf(ilWeights_ptr, "%f", inputLayerWeights->matrixData[currWeight]);
        }
        else 
        {
            fprintf(ilWeights_ptr, "%f,", inputLayerWeights->matrixData[currWeight]);
        }
    }
    INPUT_LAYER_WEIGHTS = xCreateMatrix(inputLayerWeights->rows, inputLayerWeights->cols, inputLayerWeights->matrixData);
    vReconstructInputLayer(NN, inputLayerWeights);
    fclose(ilWeights_ptr);
}

void vUpdateOutputLayerWeights (Matrix * outputLayerWeights, NerualNetwork * NN)
{
    FILE * olWeights_ptr = fopen("../../../Libaries/NeuralC/Sources/Modules/LayerCreation/OutputLayer/OutputWeights.csv", "w");

    for (uint16_t currRow = 0; currRow < outputLayerWeights->rows; currRow++)
    {
        for (uint16_t currCol = 0; currCol < outputLayerWeights->cols; currCol++)
        {
            fprintf(olWeights_ptr, "%f", outputLayerWeights->matrixData[currRow * outputLayerWeights->cols + currCol]);
            if (currCol != outputLayerWeights->cols - 1) 
            {
                fprintf(olWeights_ptr, ",");
            }
        }
        if (currRow < outputLayerWeights->rows-1)
        {
            fprintf(olWeights_ptr, "\n");
        }
    }
    vReconstructOutputWeights(NN, outputLayerWeights);
}
void vUpdateOutputLayerBiases (Matrix * outputLayerBiases, NerualNetwork * NN)
{
    FILE * olBiases_ptr = fopen("../../../Libaries/NeuralC/Sources/Modules/LayerCreation/OutputLayer/OutputBiases.csv", "w");

    for (uint16_t currRow = 0; currRow < outputLayerBiases->rows; currRow++)
    {
        for (uint16_t currCol = 0; currCol < outputLayerBiases->cols; currCol++)
        {
            fprintf(olBiases_ptr, "%f", outputLayerBiases->matrixData[currRow * outputLayerBiases->cols + currCol]);
            if (currCol != outputLayerBiases->cols - 1) 
            {
                fprintf(olBiases_ptr, ",");
            }
        }
        if (currRow < outputLayerBiases->rows-1)
        {
            fprintf(olBiases_ptr, "\n");
        }
    }
    vReconstructOutputBiases(NN, outputLayerBiases);
}

void vUpdateHiddenLayerWeights (Matrix * hiddenLayerWeights, NerualNetwork * NN, uint16_t hiddenLayerNum)
{
    char filename[100];
    sprintf(filename, "../../../Libaries/NeuralC/Sources/Modules/LayerCreation/HiddenLayers/HiddenWeights%d.csv", hiddenLayerNum);

    FILE * hlWeights_ptr = fopen(filename, "w");

    for (uint16_t currRow = 0; currRow < hiddenLayerWeights->rows; currRow++)
    {
        for (uint16_t currCol = 0; currCol < hiddenLayerWeights->cols; currCol++)
        {
            fprintf(hlWeights_ptr, "%f", hiddenLayerWeights->matrixData[currRow * hiddenLayerWeights->cols + currCol]);
            if (currCol != hiddenLayerWeights->cols - 1) 
            {
                fprintf(hlWeights_ptr, ",");
            }
        }
        if (currRow < hiddenLayerWeights->rows-1)
        {
            fprintf(hlWeights_ptr, "\n");
        }
    }

    vReconstructHiddenWeights(NN, hiddenLayerWeights, hiddenLayerNum);
}

void vUpdateHiddenLayerBiases (Matrix * hiddenLayerBiases, NerualNetwork * NN, uint16_t hiddenLayerNum)
{
    char filename[100];
    sprintf(filename, "../../../Libaries/NeuralC/Sources/Modules/LayerCreation/HiddenLayers/HiddenBiases%d.csv", hiddenLayerNum);

    FILE * hlBiases_ptr = fopen(filename, "w");

    for (uint16_t currRow = 0; currRow < hiddenLayerBiases->rows; currRow++)
    {
        for (uint16_t currCol = 0; currCol < hiddenLayerBiases->cols; currCol++)
        {
            fprintf(hlBiases_ptr, "%f", hiddenLayerBiases->matrixData[currRow * hiddenLayerBiases->cols + currCol]);
            if (currCol != hiddenLayerBiases->cols - 1) 
            {
                fprintf(hlBiases_ptr, ",");
            }
        }
        if (currRow < hiddenLayerBiases->rows-1)
        {
            fprintf(hlBiases_ptr, "\n");
        }
    }

    vReconstructHiddenBiases(NN, hiddenLayerBiases, hiddenLayerNum);
}