#ifndef LAYER_CREATION_H
#define LAYER_CREATION_H

#include "../Matrix/matrix.h"
#include <stdio.h>

extern Matrix * INPUT_LAYER_WEIGHTS;

extern Matrix * HIDDEN_LAYER_WEIGHTS[];
extern Matrix * HIDDEN_LAYER_BIASES[];

extern Matrix * OUTPUT_LAYER_WEIGHTS;
extern Matrix * OUTPUT_LAYER_BIASES;

void vCreateNNLayerMatricies (void);

#endif